import queue


def copy_queue(value):
    copy = queue.Queue()
    for i in value.queue: copy.put(i)
    return copy
