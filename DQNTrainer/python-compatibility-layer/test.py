#!/usr/bin/python

from PySharedMemoryInterface import SharedMemoryInterface
import numpy as np
import random
import time

print("Testing!")
x = SharedMemoryInterface()

def process_message(message, user_input=False):
    print(f"Id: {message.id}")
    print("Board layout: ")
    for i in range(4):
        row = (message.board >> 16*(3-i)) & 0xFFFF
        print(format(row, '016b'))

    print("Moves:","{:05b}".format(message.moves))
    if (message.moves & 16) > 0:
        return message.id,-1
    choices = [None] * 4
    for i in range(4):
        choices[i] = ((message.moves >> i) & 1) > 0
    n = -1
    if user_input:
        while True:
            n = int(input("Enter a move [1,2,3,4] => [left,right,up,down]: "))-1
            if choices[n-1]:
                break
    else:
        while True:
            n = random.randint(0,3)
            if choices[n]:
                break
    return message.id,(1 << n) # return a random valid move 

ended = [False] * 6
ended_count = 0
now = time.time()
while True:
    message, success = x.getMessage()
    if not success or ended[message.id]:
        continue
    id, response = process_message(message, False)
    print(f"Sending move: {response}")
    if response < 0:
        print("Game ended")
        ended[id] = True
        ended_count += 1
        print(f"Games ended: {ended_count}")
        if ended_count == 6:
            break
    else:
        x.putResponse(id,response)
end = time.time()
print(f"Elapsed time: {end-now} seconds")