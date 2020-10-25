import numpy as np
import pandas as pd
import math



def preprocess (wtf, ttype, userid):
    # input :
    #   1. time stamp
    #   2. x
    #   3. y
    #   4. type
    input_raw_data = np.array(wtf)
    input_raw_data[:, 0] = input_raw_data[:, 0] + 0.000001
    #print ("fuck ", len(input_raw_data))

    li = []
    this_type = []
    start = -1
    state = ''
    for i in range(0, len(input_raw_data)):
        if start == -1 :
            start = i
            state = ttype[i]
        else :
            if state != ttype[i] and ttype[i] != 'Pressed' and ttype[i] != 'Released':
                li.append(input_raw_data[start:i+1])
                this_type.append(state)
                start = -1


    #print ("this is in preprocess.py, and the length is ", len(li))
    to_res = []
    for index, input_data in enumerate(li) :
        delta_t = input_data[:, 0] - np.insert(input_data[:-1, 0], 0, 0, axis = 0)
        delta_x = input_data[:, 1] - np.insert(input_data[:-1, 1], 0, 0, axis = 0)
        delta_y = input_data[:, 2] - np.insert(input_data[:-1, 2], 0, 0, axis = 0)

        # theta
        theta = np.insert(np.arctan2(delta_y[1:], delta_x[1:]), 0, 0, axis = 0)


        # vx
        v_x = np.insert(delta_x[1:] / delta_t[1:], 0, 0, axis = 0)
        v_y = np.insert(delta_y[1:] / delta_t[1:], 0, 0, axis = 0)
        v = np.sqrt(v_x ** 2 + v_y ** 2)

        delta_sum = np.sqrt(delta_x**2 + delta_y ** 2)

        # trajectory
        s = np.zeros(len(input_data))
        for i in range(1, len(input_data)) :
            s[i] = s[i-1] + delta_sum[i-1]

        # c
        c = np.insert(theta[1:] / s[1:], 0, 0, axis = 0)

        # a
        delta_v = v - np.insert(v[:-1], 0, 0, axis = 0)
        a = np.insert(delta_v[1:] / delta_t[1:], 0, 0, axis = 0)

        # j
        delta_a = a - np.insert(a[:-1], 0, 0, axis = 0)
        j = np.insert(delta_a[1:] / delta_t[1:], 0, 0, axis = 0)

        # elapse time
        elapse_time = input_data[-1, 0] - input_data[0, 0]

        # trajectory length
        trajectory_length = s[-1]

        # distance end to end 
        distance_end_to_end = math.sqrt((input_data[-1, 1] - input_data[0, 1]) ** 2 +  (input_data[-1, 1] - input_data[0, 1]) ** 2)

        # straightness
        straightness = distance_end_to_end / trajectory_length

        num_point = len(input_data)

        sum_of_angle = np.sum(theta)

        th = 0.05
        sharp_angle = len(theta < th)

        x1 = input_data[0, 1]
        x2 = input_data[-1, 1]
        y1 = input_data[0, 2]
        y2 = input_data[-1, 2]

        # direction 
        fuck_len = math.sqrt((x2-x1)**2 + (y2-y1) ** 2)
        if y2 - y1 < 0 :
            # down
            my_arc_value = math.acos((y2-y1)/fuck_len)
            direction = int((360 - my_arc_value * (180 / math.pi)) / 45) + 1
        elif y2 - y1 < 0.00001:
            direction = 0
        else :
            # up
            my_arc_value = math.acos((y2-y1)/fuck_len)
            direction = int((my_arc_value * (180 / math.pi)) / 45) + 1


        fuck_down = math.sqrt((y1-y2) ** 2 + (x2-x1) ** 2)
        fuck = ((y1-y2)*input_data[:, 1] + (x2-x1)*input_data[:, 2] + x1*(y2-y1) + y1*(x1-x2)) / fuck_down
        largest_divation = np.max(fuck)

#        fucking_type = this_type[index]
        if this_type[index] == 'Drag' :
            fucking_type = 1
        else :
            fucking_type = 2
        res = [np.mean(v_x), np.std(v_x), np.min(v_x), np.max(v_x), 
                np.mean(v_y), np.std(v_y), np.min(v_y), np.max(v_y),
                np.mean(v), np.std(v), np.min(v), np.max(v),
                np.mean(a), np.std(a), np.min(a), np.max(a),
                np.mean(j), np.std(j), np.min(j), np.max(j),
                np.mean(theta), np.std(theta), np.min(theta), np.max(theta),
                np.mean(c), np.std(c), np.min(c), np.max(c),
                fucking_type, 
                elapse_time, 
                trajectory_length, 
                distance_end_to_end, 
                direction,
                straightness, 
                num_point, 
                sum_of_angle, 
                largest_divation, 
                sharp_angle, 
                userid]
        for k in range(len(res)) :
            if res[k] != res[k] or res[k] == float("inf") or res[k] == float("-inf"):
                res[k] = 0
        to_res.append(res)

    return to_res



def get_user_data(file_name1, file_name2='none'):
    original_user = None
    with open(file_name1, "r") as fh:
        l = fh.readlines()
        li = []
        ttype = []
        for i in l :
            temp = i.split(',')
            t = float(temp[0])
            x = float(temp[1])
            y = float(temp[2])
            t_type = str(temp[3][1:-1])
            li.append([t, x, y])
            ttype.append(t_type)
        original_user = preprocess(li, ttype, 1)

    if file_name2 != 'none' :
        other_user = None
        with open(file_name2, "r") as fh:
            l = fh.readlines()
            li = []
            ttype = []
            for i in l :
                temp = i.split(',')
                t = float(temp[0])
                x = float(temp[1])
                y = float(temp[2])
                t_type = str(temp[3][1:-1])
                li.append([t, x, y])
                ttype.append(t_type)
            other_user = preprocess(li, ttype, 2)

        original = pd.DataFrame(original_user)
        other = pd.DataFrame(other_user)
        fuck = pd.concat([original, other], axis = 0) 
        #fuck.to_csv('testest.csv', index=False)
        return fuck
    
    return pd.DataFrame(original_user)


#get_user_data('mouse_log', 1)
