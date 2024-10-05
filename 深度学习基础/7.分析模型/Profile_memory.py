from memory_profiler import profile

@profile
def my_function():
    return dataset.__getitem__(5)

if __name__ == "__main__":
    my_function()