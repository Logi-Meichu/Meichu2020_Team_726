from pynput import mouse, keyboard
import keyboard as kb
import time
import sys
from contextlib import redirect_stdout
from getkey import getkey, keys

PRINT_MODE_ON = False

FILEPATH = 'mouse_log.txt'
t_start = 0.0
t_end = 0.0
delta_t = 0.0

mouse_time_end = 0.0
mouse_initial = 0.0

action = []  # for keyboard


def redirect(filePath, val):
    with open(filePath, 'a') as out:
        with redirect_stdout(out):
            print(val)


def on_move(x, y):
    global mouse_initial
    mouse_time_end = time.time()
    meas_time = mouse_time_end - mouse_initial
    if PRINT_MODE_ON == True:
        print(f'{meas_time}, {x}, {y}, Moved')
    redirect(FILEPATH, f'{meas_time}, {x}, {y}, Moved')


def on_click(x, y, button, pressed):
    global mouse_initial
    mouse_time_end = time.time()
    meas_time = mouse_time_end - mouse_initial
    if PRINT_MODE_ON == True:
        print('{}, {}, {}, {}'.format(meas_time, x, y,
        'Pressed' if pressed else 'Released'))
    redirect(FILEPATH, '{}, {}, {}, {}'.format(meas_time, x, y,
                                               'Pressed' if pressed else 'Released'))
    # if getkey(blocking = False) == 'ESC':
    #     # Stop listener
    #     return False

# def on_scroll(x, y, dx, dy):
#     print('Scrolled {0} at {1}'.format(
#         'down' if dy < 0 else 'up',
#         (x, y)))


def on_press(key):
    global t_start
    try:
        t_start = time.time()
        if PRINT_MODE_ON == True:
            print('alphanumeric key {0} pressed'.format(key.char))
        action.append('alphanumeric key {0} pressed\n'.format(key.char))
    except AttributeError:
        t_start = time.time()
        if PRINT_MODE_ON == True:
            print('special key {0} pressed'.format(key))
        action.append('special key {0} pressed\n'.format(key))


def on_release(key):
    global t_end, delta_t
    t_end = time.time()
    delta_t = t_end - t_start
    # print('fuck')
    if PRINT_MODE_ON == True:

        print('{0} released'.format(key))
        print(f'duration = {delta_t}')
    action.append('{0} released\n'.format(key))
    action.append(f'duration = {delta_t:4f}\n\n')
    if key == keyboard.Key.f8:
        # Stop listener
        with open('keyboard.log', 'w') as fh:
            fh.writelines(action)
        print('Collect data ends!')
        return False


# Collect events until released
# with mouse.Listener(
#         on_move=on_move,
#         on_click=on_click,
#         on_scroll=on_scroll) as listener:
#     listener.join()

# ...or, in a non-blocking fashion:
# if __name__ == '__main__':
def main():
    global mouse_initial
    print('Press letter "S" to start collecting data')
    mouse_initial = time.time()
    
    # Clear the mouse data
    with open('mouse_log.txt', 'w') as f:
        f.write('\n')

    # detect if button s is pressed
    while True:
        if kb.is_pressed('s') or kb.is_pressed('S'): 
            break

    global listener
    listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click)
    listener.start()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as key_listener:
        key_listener.join()


main()
with open('mouse_log.txt', 'r') as fh:
    L = fh.readlines()
    # print(L)
    new = []
    press = False
    offset = float(L[1].split(',')[0])
    # print(offset)
    for action in L[1:]:
        temp = action.split(',')
        # print(temp[0])
        temp[0] = str(float(temp[0]) - offset)
        if temp[3] == ' Pressed\n':
            press = True
        if press and temp[3] == ' Moved\n':
            temp[3] = ' Drag\n'
        if temp[3] == ' Released\n':
            press = False
        new.append(','.join(temp))
    with open('new_mouse_log.txt', 'w') as fh2:
        fh2.writelines(new)
    print('Parse the data successfully!')


