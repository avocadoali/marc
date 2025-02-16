import itertools
import random
from typing import List

import numpy as np
from torchtune.data import arc_to_messages

from arclib.arc import Task
from arclib.augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from arclib.messagers import MessageRepresenter


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    print("Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply))
    return augmenters_to_apply


def format_and_filter(formatter, tokenizer, task, train_on_input: False):
    task = formatter.encode(task)
    data = {"input": task[0], "output": task[1]}
    messages = arc_to_messages(data, train_on_input=False)
    tokens, labels = tokenizer.tokenize_messages(messages)
    data["total_tokens"] = len(tokens)
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:

    rng = np.random.RandomState(seed)

    # print('len(augmenters) in get_test_time_train_data: ', len(augmenters))
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]


        # breakpoint()
        for comb in combs:
            # breakpoint()
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )
        
        

    # print(f"combs: {len(combs)}")
    # print(f"initial_tasks: {len(initial_tasks)}")

    augmented_tasks = []
    iterations = 0
    skipped = 0
    
    rng = np.random.RandomState(rng.randint(0, 2**32))
    for augmenter in augmenters:
        for task in initial_tasks:
            iterations += 1
            rng = np.random.RandomState(rng.randint(0, 2**32))
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                skipped += 1
                continue
            augmented_tasks.append(task)
    if skipped > 0:
        print(f"Skipped, grid too large: {skipped} tasks")

    print(f'augmenters: {len(augmenters)}')
    print(f"iterations: {iterations}")
    print(f"permute_n: {permute_n}")
    print(f"Duplicates first: {len(augmented_tasks) - len(set(augmented_tasks))}")
    augmented_tasks = list(set(augmented_tasks + initial_tasks))
    print(f"augmented_tasks: {len(augmented_tasks)}")

    color_and_permute_augmented_tasks = []

    # breakpoint()
    
    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)


    # breakpoint()
    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks
    print(f"augmented_tasks + permute_augmented_tasks: {len(augmented_tasks)}")
    print(f"Duplicates second: {len(augmented_tasks) - len(set(augmented_tasks))}")
    augmented_tasks = list(set(augmented_tasks))
    print(f'len(augmented_tasks) after set: {len(augmented_tasks)}')

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    # max_tokens: int = 8192,
    # max_tokens: int = 15000,
    max_tokens: int = 20_000,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    # rng = np.random.RandomState(seed)
    # train_data_1 = get_test_time_train_data(
    #     task, augmenters, n=leave_n, permute_n=permute_n, seed=rng.randint(0, 2**32)
    # )

    # # check if train_data_1 is the same as train_data
    # t  = train_data + train_data_1
    # count = len(t) - len(set(t))
    # # print(f'len(train_data): {len(train_data)}')
    # print(f'len(train_data_1): {len(train_data_1)}')
    # print(f'same: {count}')
    # breakpoint()

    formatted_data = []
    n_filtered = 0
    token_sizes = []
    max_token_size = 0
    for task in train_data:
        formatted = format_and_filter(formatter, tokenizer, task, train_on_input=False)
        if formatted["total_tokens"] > max_token_size:
            max_token_size = formatted["total_tokens"]
        if formatted["total_tokens"] < max_tokens:
            formatted_data.append(formatted)
        else:
            n_filtered += 1
            token_sizes.append(formatted["total_tokens"])
    # breakpoint()

    print(f"Filtered bc too many tokens needed: {n_filtered}")
    # print highest 5 token sizes
    print(f"Highest 5 token sizes: {sorted(token_sizes, reverse=True)[:5]}")
    print(f"Max token size: {max_token_size}")
    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)

    # # duplicate task if it has less than 2 examples
    examples_to_add = 5 - len(task.train_examples)
    
    # leave_1_train_data_before = get_formatted_data(
    #     task, augmenters, formatter, tokenizer, leave_n=1, permute_n=permute_n, seed=seed
    # )

    # leave_2_train_data_before = get_formatted_data(
    #     task, augmenters, formatter, tokenizer, leave_n=2, permute_n=permute_n, seed=seed
    # )



    if examples_to_add > 0:

        basic_augmenters = get_augmenters(include_basic=True, include_size=False, include_chain=False, include_repeat=False, include_concat=False, )
        tasks_add = get_test_time_train_data(
            task, basic_augmenters, n=1, permute_n=permute_n, seed=seed
        )
        # randomly add 3 examples from tasks_add to task.train_examples
        sub = random.sample(tasks_add, examples_to_add)
        # breakpoint()
        for x in sub:
            task.train_examples.append(x.train_examples[0])


    print(f'initial examples new: {len(task.train_examples)}')

    # breakpoint()
    # print('len(augmenters) in get_formatted_data: ', len(augmenters))
    # breakpoint()
    print('')
    print('leave_1_train_data')
    leave_1_train_data = get_formatted_data(
        task, augmenters, formatter, tokenizer, leave_n=1, permute_n=permute_n, seed=seed
    )

    # breakpoint()

    # breakpoint()
    print('')
    print('leave_2_train_data')
    leave_2_train_data = get_formatted_data(
        task, augmenters, formatter, tokenizer, leave_n=2, permute_n=permute_n, seed=seed
    )

    # breakpoint()
    # print('task: ', task)
    # print('leave_1_train_data: ', leave_1_train_data)
    # leave_1_1_train_data = get_formatted_data(
    #     task, augmenters, formatter, tokenizer, leave_n=1, permute_n=permute_n, seed=rng.randint(0, 2**32)
    # )

    # leave_2_1_train_data = get_formatted_data(
    #     task, augmenters, formatter, tokenizer, leave_n=2, permute_n=permute_n, seed=rng.randint(0, 2**32)
    # )

    # same_count = 0
    # for idx, x in enumerate(leave_1_1_train_data):
    #     if leave_1_1_train_data[idx] == leave_1_train_data[idx]:
    #         same_count += 1
    # # print(f'same_count_leave_1_1_train_data: {same_count}')


    # same_count = 0
    # for idx, x in enumerate(leave_2_1_train_data):
    #     if leave_2_1_train_data[idx] == leave_2_train_data[idx]:
    #         same_count += 1

    # print(f'same_count_leave_2_1_train_data: {same_count}')

    # leave_3_train_data = get_formatted_data(
    #     task, augmenters, formatter, tokenizer, leave_n=3, permute_n=permute_n, seed=seed
    # )

    # train = leave_1_train_data

    print(f"leave_1_train_data: {len(leave_1_train_data)}")
    # print(f"leave_1_1_train_data: {len(leave_1_1_train_data)}")
    print(f"leave_2_train_data: {len(leave_2_train_data)}")
    # print(f"leave_2_1_train_data: {len(leave_2_1_train_data)}")
    # print(f"leave_3_train_data: {len(leave_3_train_data)}")
    print('total train data: ', len(leave_1_train_data) + len(leave_2_train_data))
    print('================================================================================')
    # print('================================================================================')
    # print('================================================================================')
    # # if len(train) == 0:
    #     train = leave_2_train_data
    # elif len(train) < Nmax:
    #     train += leave_2_train_data[: Nmax - len(train)]
    # elif len(train) > Nmax:
    #     rng.shuffle(train)
    #     train = train[:Nmax]

    # train = leave_1_train_data + leave_1_1_train_data + leave_2_train_data + leave_2_1_train_data + leave_3_train_data
    train = leave_1_train_data +  leave_2_train_data 
    
    if len(train) < Nmax:
        print(f'len(train): {len(train)} 1.1 < Nmax: {Nmax}')

        leave_1_1_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=1, permute_n=permute_n, seed=rng.randint(0, 2**32)
        )
        print(f'len(leave_1_1_train_data): {len(leave_1_1_train_data)}')
        train += leave_1_1_train_data
    
    if len(train) < Nmax:
        print(f'len(train): {len(train)} 2.1 < Nmax: {Nmax}')
        leave_2_1_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=2, permute_n=permute_n, seed=rng.randint(0, 2**32)
        )
        print(f'len(leave_2_1_train_data): {len(leave_2_1_train_data)}')
        train += leave_2_1_train_data

    #     print(f'len(leave_1_1_train_data): {len(leave_1_1_train_data)}')
    #     print(f'len(leave_2_1_train_data): {len(leave_2_1_train_data)}')
    #     train = train + leave_1_1_train_data + leave_2_1_train_data

    
    print(f'len(train): {len(train)}, initial examples: {len(task.train_examples)}')
    if len(train) > Nmax:
        train = train[:Nmax]

    print(f'len(train): {len(train)}, initial examples: {len(task.train_examples)}')
    return train
