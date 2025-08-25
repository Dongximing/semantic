import re 
import random
PROMPT = "Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. {prompt}"

def get_multiple_choice_answer(pred: str):
    """Attempts to parse the letter A, B, C, or D from the string.

    If it can't, it returns the string in raw form."""
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def get_multiple_choice_answers(data):
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(answers)

        # Map options to letters
        options = ["A", "B", "C", "D"]
        options_to_answers = {
            letter: answer for letter, answer in zip(options, answers)
        }

        # Format the options into the string
        multiple_choice_string = ", ".join(
            f"{letter}) {options_to_answers[letter]}" for letter in options
        )

        # Save the letter corresponding to the correct answer
        correct_answer_letter = next(
            letter
            for letter, answer in options_to_answers.items()
            if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter
def generate_prompt(problem):
    multiple_choice_string, correct_answer_letter = (
        get_multiple_choice_answers(problem)
    )
    problem["Answer"] = correct_answer_letter
    problem["prompt"] = problem["Question"] + "\n" + multiple_choice_string
    return PROMPT.format(
        prompt=problem["prompt"]
    )
def get_GPQA_multiple_choice_answers(data):
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(answers)

    # Map options to letters
    options = ["A", "B", "C", "D"]
    options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    # Format the options into the string
    multiple_choice_string = ", ".join(
        f"{letter}) {options_to_answers[letter]}" for letter in options
    )

    # Save the letter corresponding to the correct answer
    correct_answer_letter = next(
        letter
        for letter, answer in options_to_answers.items()
        if answer == data["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter