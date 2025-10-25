"""
Evaluate the Chat model.
All the generic code lives here, and all the evlauation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import re
import argparse
from functools import partial

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.device import get_autocast_kwargs

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# Formatting helpers for generative evaluations

FORMAT_BOXED_RE = re.compile(r"\\boxed\s*{[^{}]+}")


def completion_has_think_markers(text: str) -> bool:
    """Return True if both <think> and </think> markers appear in the completion."""
    return "<think>" in text and "</think>" in text


def completion_has_boxed_answer(text: str) -> bool:
    """Return True if the completion includes a \\boxed{} span."""
    return bool(FORMAT_BOXED_RE.search(text))

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    format_counts = {
        "total": 0,
        "think": 0,
        "boxed": 0,
        "both": 0,
    }
    correct_format_records = []
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Track formatting coverage per completion
        for sample_idx, completion in enumerate(completions):
            has_think = completion_has_think_markers(completion)
            has_box = completion_has_boxed_answer(completion)
            format_counts["total"] += 1
            format_counts["think"] += int(has_think)
            format_counts["boxed"] += int(has_box)
            format_counts["both"] += int(has_think and has_box)
            if has_think and has_box:
                correct_format_records.append((i, sample_idx))

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Gather and log completions that followed the desired format
    if correct_format_records:
        prefix = f"[FORMAT] Rank {ddp_rank}"
        for problem_idx, sample_idx in correct_format_records:
            print(f"{prefix} | Problem {problem_idx} sample {sample_idx}: ✅ contains <think>...</think> and \\boxed{{}} answer", flush=True)

    # Aggregate formatting statistics across all ranks
    fmt_tensor = torch.tensor(
        [
            format_counts["think"],
            format_counts["boxed"],
            format_counts["both"],
            format_counts["total"],
        ],
        dtype=torch.long,
        device=device,
    )
    if ddp:
        dist.all_reduce(fmt_tensor, op=dist.ReduceOp.SUM)
    think_total, boxed_total, both_total, total_completions = fmt_tensor.tolist()
    if total_completions > 0:
        think_ratio = think_total / total_completions
        boxed_ratio = boxed_total / total_completions
        both_ratio = both_total / total_completions
    else:
        think_ratio = boxed_ratio = both_ratio = 0.0
    print0(
        f"Format coverage: <think> tags {think_total}/{total_completions} ({100*think_ratio:.2f}%), "
        f"\\boxed{{}} answers {boxed_total}/{total_completions} ({100*boxed_ratio:.2f}%), "
        f"both {both_total}/{total_completions} ({100*both_ratio:.2f}%)"
    )

    format_metrics = {
        "total_completions": total_completions,
        "think_tagged": think_total,
        "boxed_answer": boxed_total,
        "both_present": both_total,
        "think_tagged_ratio": think_ratio,
        "boxed_answer_ratio": boxed_ratio,
        "both_present_ratio": both_ratio,
    }

    # Return the accuracy together with formatting metrics
    return num_passed/total, format_metrics

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the avilable letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    # Create the evaluation object
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == 'generative':
        acc, format_metrics = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
        format_metrics = None
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc, format_metrics

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|mid|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_kwargs = get_autocast_kwargs(device)
    autocast_ctx = torch.amp.autocast(**{**autocast_kwargs, "dtype": ptdtype})

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Run all the task evaluations sequentially
    results = {}
    format_metrics_by_task = {}
    for task_name in task_names:
        with autocast_ctx:
            acc, format_metrics = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")
            if format_metrics is not None:
                format_metrics_by_task[task_name] = format_metrics

    # Log to report
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    payload = [
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
    ]
    if format_metrics_by_task:
        payload.append({"format_metrics": format_metrics_by_task})
    get_report().log(section="Chat evaluation " + args.source, data=payload)

    compute_cleanup()
