import numpy as np


def main_diag(left, right):
    unit = np.zeros((5, 8), 'str')
    unit[0][0] = left
    for i in range(2):
        unit[1][i] = left
    for i in range(3):
        unit[2][i] = left
    for i in range(5):
        unit[3][i] = left
    for i in range(6):
        unit[4][i] = left

    unit[0][1] = '.'
    unit[1][2] = '.'
    unit[2][3] = '.'
    unit[2][4] = '.'
    unit[3][5] = '.'
    unit[4][6] = '.'

    for i in range(5):
        for j in range(8):
            if unit[i][j] != left and unit[i][j] != '.':
                unit[i][j] = right

    for i in range(5):
        unit[i][0] = '.'

    return unit


def main_diag2(left, right):
    unit = np.zeros((5, 8), 'str')
    unit[0][0] = left
    for i in range(2):
        unit[1][i] = left
    for i in range(3):
        unit[2][i] = left
    for i in range(5):
        unit[3][i] = left
    for i in range(6):
        unit[4][i] = left

    unit[0][1] = '.'
    unit[1][2] = '.'
    unit[2][3] = '.'
    unit[2][4] = '.'
    unit[3][5] = '.'
    unit[4][6] = '.'

    for i in range(5):
        for j in range(8):
            if unit[i][j] != left and unit[i][j] != '.':
                unit[i][j] = right

    return unit


def sub_diag(left, right):
    unit = np.zeros((5, 8), 'str')
    for i in range(6):
        unit[0][i] = left
    for i in range(5):
        unit[1][i] = left
    for i in range(3):
        unit[2][i] = left
    for i in range(2):
        unit[3][i] = left
    for i in range(1):
        unit[4][i] = left

    unit[0][6] = '.'
    unit[1][5] = '.'
    unit[2][3] = '.'
    unit[2][4] = '.'
    unit[3][2] = '.'
    unit[4][1] = '.'
    for i in range(5):
        for j in range(8):
            if unit[i][j] != left and unit[i][j] != '.':
                unit[i][j] = right
    for i in range(5):
        unit[i][0] = '.'

    return unit


def sub_diag2(left, right):
    unit = np.zeros((5, 8), 'str')
    for i in range(6):
        unit[0][i] = left
    for i in range(5):
        unit[1][i] = left
    for i in range(3):
        unit[2][i] = left
    for i in range(2):
        unit[3][i] = left
    for i in range(1):
        unit[4][i] = left

    unit[0][6] = '.'
    unit[1][5] = '.'
    unit[2][3] = '.'
    unit[2][4] = '.'
    unit[3][2] = '.'
    unit[4][1] = '.'
    for i in range(5):
        for j in range(8):
            if unit[i][j] != left and unit[i][j] != '.':
                unit[i][j] = right
    return unit


def whole(color):
    unit = np.zeros((5, 8), 'str')
    for i in range(5):
        for j in range(8):
            unit[i][j] = color
    for i in range(5):
        unit[i][0] = '.'

    return unit


def lr(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
    unit = a1
    unit = np.append(unit, a2, axis=1)
    unit = np.append(unit, a3, axis=1)
    unit = np.append(unit, a4, axis=1)
    unit = np.append(unit, a5, axis=1)
    unit = np.append(unit, a6, axis=1)
    unit = np.append(unit, a7, axis=1)
    unit = np.append(unit, a8, axis=1)
    unit = np.append(unit, a9, axis=1)
    unit = np.append(unit, a10, axis=1)
    unit = np.append(unit, a11, axis=1)
    unit = np.append(unit, a12, axis=1)
    return unit


def ud(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
    unit = a1
    unit = np.append(unit, a2, axis=0)
    unit = np.append(unit, a3, axis=0)
    unit = np.append(unit, a4, axis=0)
    unit = np.append(unit, a5, axis=0)
    unit = np.append(unit, a6, axis=0)
    unit = np.append(unit, a7, axis=0)
    unit = np.append(unit, a8, axis=0)
    unit = np.append(unit, a9, axis=0)
    unit = np.append(unit, a10, axis=0)
    unit = np.append(unit, a11, axis=0)
    unit = np.append(unit, a12, axis=0)
    return unit


def cuber_map():
    a1_1 = main_diag('j', '.')
    a1_2 = whole('.')
    a1_3 = whole('.')
    a1_4 = whole('.')
    a1_5 = whole('.')
    a1_6 = sub_diag('.', 'a')
    a1_7 = main_diag2('a', '.')
    a1_8 = whole('.')
    a1_9 = whole('.')
    a1_10 = whole('.')
    a1_11 = whole('.')
    a1_12 = sub_diag('.', 'D')
    a1 = lr(a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, a1_8, a1_9, a1_10, a1_11, a1_12)

    a2_1 = whole('j')
    a2_2 = main_diag('k', '.')
    a2_3 = whole('.')
    a2_4 = whole('.')
    a2_5 = sub_diag('.', 'd')
    a2_6 = main_diag2('d', 'a')
    a2_7 = sub_diag2('a', 'b')
    a2_8 = main_diag2('b', '.')
    a2_9 = whole('.')
    a2_10 = whole('.')
    a2_11 = sub_diag('.', 'C')
    a2_12 = whole('D')
    a2 = lr(a2_1, a2_2, a2_3, a2_4, a2_5, a2_6, a2_7, a2_8, a2_9, a2_10, a2_11, a2_12)

    a3_1 = main_diag('m', 'j')
    a3_2 = whole('k')
    a3_3 = main_diag('l', '.')
    a3_4 = sub_diag('.', 'g')
    a3_5 = main_diag2('g', 'd')
    a3_6 = sub_diag2('d', 'e')
    a3_7 = main_diag2('e', 'b')
    a3_8 = sub_diag2('b', 'c')
    a3_9 = main_diag2('c', '.')
    a3_10 = sub_diag('.', 'B')
    a3_11 = whole('C')
    a3_12 = sub_diag('D', 'G')
    a3 = lr(a3_1, a3_2, a3_3, a3_4, a3_5, a3_6, a3_7, a3_8, a3_9, a3_10, a3_11, a3_12)

    a4_1 = whole('m')
    a4_2 = main_diag('n', 'k')
    a4_3 = whole('l')
    a4_4 = main_diag('s', 'g')
    a4_5 = sub_diag2('g', 'h')
    a4_6 = main_diag2('h', 'e')
    a4_7 = sub_diag2('e', 'f')
    a4_8 = main_diag2('f', 'c')
    a4_9 = sub_diag2('c', '4')
    a4_10 = whole('B')
    a4_11 = sub_diag('C', 'F')
    a4_12 = whole('G')
    a4 = lr(a4_1, a4_2, a4_3, a4_4, a4_5, a4_6, a4_7, a4_8, a4_9, a4_10, a4_11, a4_12)

    a5_1 = main_diag('p', 'm')
    a5_2 = whole('n')
    a5_3 = main_diag('o', 'l')
    a5_4 = whole('s')
    a5_5 = main_diag('t', 'h')
    a5_6 = sub_diag2('h', 'i')
    a5_7 = main_diag2('i', 'f')
    a5_8 = sub_diag2('f', '3')
    a5_9 = whole('4')
    a5_10 = sub_diag('B', 'E')
    a5_11 = whole('F')
    a5_12 = sub_diag('G', 'J')
    a5 = lr(a5_1, a5_2, a5_3, a5_4, a5_5, a5_6, a5_7, a5_8, a5_9, a5_10, a5_11, a5_12)

    a6_1 = whole('p')
    a6_2 = main_diag('q', 'n')
    a6_3 = whole('o')
    a6_4 = main_diag('v', 's')
    a6_5 = whole('t')
    a6_6 = main_diag('u', 'i')
    a6_7 = sub_diag2('i', '2')
    a6_8 = whole('3')
    a6_9 = sub_diag('4', '7')
    a6_10 = whole('E')
    a6_11 = sub_diag('F', 'I')
    a6_12 = whole('J')
    a6 = lr(a6_1, a6_2, a6_3, a6_4, a6_5, a6_6, a6_7, a6_8, a6_9, a6_10, a6_11, a6_12)

    a7_1 = main_diag('.', 'p')
    a7_2 = whole('q')
    a7_3 = main_diag('r', 'o')
    a7_4 = whole('v')
    a7_5 = main_diag('w', 't')
    a7_6 = whole('u')
    a7_7 = whole('2')
    a7_8 = sub_diag('3', '6')
    a7_9 = whole('7')
    a7_10 = sub_diag('E', 'H')
    a7_11 = whole('I')
    a7_12 = sub_diag('J', '.')
    a7 = lr(a7_1, a7_2, a7_3, a7_4, a7_5, a7_6, a7_7, a7_8, a7_9, a7_10, a7_11, a7_12)

    a8_1 = whole('.')
    a8_2 = main_diag('.', 'q')
    a8_3 = whole('r')
    a8_4 = main_diag('y', 'v')
    a8_5 = whole('w')
    a8_6 = main_diag('x', 'u')
    a8_7 = sub_diag('2', '5')
    a8_8 = whole('6')
    a8_9 = sub_diag('7', 'A')
    a8_10 = whole('H')
    a8_11 = sub_diag('I', '.')
    a8_12 = whole('.')
    a8 = lr(a8_1, a8_2, a8_3, a8_4, a8_5, a8_6, a8_7, a8_8, a8_9, a8_10, a8_11, a8_12)

    a9_1 = whole('.')
    a9_2 = whole('.')
    a9_3 = main_diag('.', 'r')
    a9_4 = whole('y')
    a9_5 = main_diag('z', 'w')
    a9_6 = whole('x')
    a9_7 = whole('5')
    a9_8 = sub_diag('6', '9')
    a9_9 = whole('A')
    a9_10 = sub_diag('H', '.')
    a9_11 = whole('.')
    a9_12 = whole('.')
    a9 = lr(a9_1, a9_2, a9_3, a9_4, a9_5, a9_6, a9_7, a9_8, a9_9, a9_10, a9_11, a9_12)

    a10_1 = whole('.')
    a10_2 = whole('.')
    a10_3 = whole('.')
    a10_4 = main_diag('.', 'y')
    a10_5 = whole('z')
    a10_6 = main_diag('1', 'x')
    a10_7 = sub_diag('5', '8')
    a10_8 = whole('9')
    a10_9 = sub_diag('A', '.')
    a10_10 = whole('.')
    a10_11 = whole('.')
    a10_12 = whole('.')
    a10 = lr(a10_1, a10_2, a10_3, a10_4, a10_5, a10_6, a10_7, a10_8, a10_9, a10_10, a10_11, a10_12)

    a11_1 = whole('.')
    a11_2 = whole('.')
    a11_3 = whole('.')
    a11_4 = whole('.')
    a11_5 = main_diag('.', 'z')
    a11_6 = whole('1')
    a11_7 = whole('8')
    a11_8 = sub_diag('9', '.')
    a11_9 = whole('.')
    a11_10 = whole('.')
    a11_11 = whole('.')
    a11_12 = whole('.')
    a11 = lr(a11_1, a11_2, a11_3, a11_4, a11_5, a11_6, a11_7, a11_8, a11_9, a11_10, a11_11, a11_12)

    a12_1 = whole('.')
    a12_2 = whole('.')
    a12_3 = whole('.')
    a12_4 = whole('.')
    a12_5 = whole('.')
    a12_6 = main_diag('.', '1')
    a12_7 = sub_diag('8', '.')
    a12_8 = whole('.')
    a12_9 = whole('.')
    a12_10 = whole('.')
    a12_11 = whole('.')
    a12_12 = whole('.')
    a12 = lr(a12_1, a12_2, a12_3, a12_4, a12_5, a12_6, a12_7, a12_8, a12_9, a12_10, a12_11, a12_12)

    a = ud(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)

    return a
