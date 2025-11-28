#!/usr/bin/python

from PySharedMemoryInterface import SharedMemoryInterface
import numpy as np
import random
import time

message_dtype = np.dtype([
    ('id', np.uint8),
    ('board', np.uint64),
    ('moves', np.uint8),
    ('reward', np.double)
])

def print_board(board):
    board_list = []
    for i in range(16):
        cell = (board >> 4*(15-i)) & 0x000F
        cell = 0 if cell == 0 else 2 ** cell
        board_list.append(cell)
    board_matrix = []
    for i in range(4):
        ind = 4*i
        board_matrix.append(board_list[ind:ind+4])
    for row in board_matrix:
        print(row)


def process_message(message, user_input=False):
    if user_input:
        print(f"\r\033[{9}A", end="")
        print(f"Id: {message["id"]}")
        print("Board layout: ")
        print_board(message["board"])

    if (message["moves"] & 16) > 0:
        return message["id"],-1
    if user_input:
        print("Moves:","{:05b}".format(message["moves"]))
        print("Enter a move [1,2,3,4] => [left,right,up,down]: ",end='')
        while True:
            n = (1 << (int(input())-1))
            if (message["moves"] & n):
                break
            else:
                print(f"\r\033[{1}A", end="")
    else:
        while True:
            n = (1 << random.randint(0,3))
            if (message["moves"] & n):
                break
    return message["id"],n # return a random valid move 


x = SharedMemoryInterface()
process_count = 16
user_input = False

ended = [False] * process_count
ended_count = 0
now = time.time()
loop = True
while loop:
    message_batch_buf = x.getMessageBatch()
    if message_batch_buf is None:
        continue
    messages = np.frombuffer(message_batch_buf, dtype=message_dtype)
    for message in messages:
        if ended[message["id"]]:
            continue
        id, response = process_message(message, user_input)
        if user_input:
            print(f"Sending move: {format(response,"04b")}")
        if response < 0:
            ended[id] = True
            ended_count += 1
            print(f"Games ended: {ended_count}")
            print(ended)
            if user_input:
                print(f"Games ended: {ended_count}")
            if ended_count == process_count:
                loop = False
                break
        else:
            x.putResponse(id,response)
end = time.time()
print(f"Elapsed time: {end-now} seconds"))