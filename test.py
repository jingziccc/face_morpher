i = 5
def test():
    global i
    i = 6
    print(i)
    
def test2(i_list):
    i = i_list[0]
    i = 2
    
if __name__ == '__main__':
    i_list = [1]
    test2(i_list)
    print(i_list[0])