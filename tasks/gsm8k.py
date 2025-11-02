"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import Dataset, load_dataset
from tasks.common import Task


BOXED_RE = re.compile(r"\\boxed\s*{([^{}]+)}")
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """
    Extract the numerical answer from a ``\\boxed{...}`` span. Returns ``None``
    when no boxed answer is present.
    """
    match = BOXED_RE.search(completion)
    if not match:
        return None
    match_str = match.group(1).strip()
    match_str = match_str.replace(",", "")
    return match_str


THINK_TAG_BONUS = 0.1
PARSED_BOX_BONUS = 0.1


class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds: Dataset = load_dataset("accountblabla/gsm8k-sorted", subset, split=split) # type: ignore

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question']
        answer = row['answer']
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Convert the final answer marker to the \boxed{} convention used by RL.
        if assistant_message_parts and assistant_message_parts[-1]["type"] == "text":
            last_text = assistant_message_parts[-1]["text"]
            match = GSM_RE.search(last_text)
            if match:
                answer_str = match.group(1).strip()
                before = last_text[:match.start()]
                after = last_text[match.end():]
                assistant_message_parts[-1] = {
                    "type": "text",
                    "text": f"{before}\\boxed{{{answer_str}}}{after}",
                }
        # No put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {"messages": messages}
        return conversation

    def evaluate(self, problem, completion):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(completion, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = problem['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(completion)
        if ref_num is None or pred_num is None:
            return 0
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        reward = float(is_correct)
        # Encourage reasoning traces and well-formed boxed answers even when incorrect.
        if "<think>" in assistant_response and "</think>" in assistant_response:
            reward += THINK_TAG_BONUS
        if extract_answer(assistant_response) is not None:
            reward += PARSED_BOX_BONUS
        return reward
