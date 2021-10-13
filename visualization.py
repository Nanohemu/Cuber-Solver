from PIL import Image
import matplotlib.pyplot as plt
from visual_utils import *

POS_DICT = {
    # '.': '-',

    'a': '000', 'b': '001', 'c': '002',
    'd': '010', 'e': '011', 'f': '012',
    'g': '020', 'h': '021', 'i': '022',

    'j': '100', 'k': '101', 'l': '102',
    'm': '110', 'n': '111', 'o': '112',
    'p': '120', 'q': '121', 'r': '122',

    's': '200', 't': '201', 'u': '202',
    'v': '210', 'w': '211', 'x': '212',
    'y': '220', 'z': '221', '1': '222',

    '2': '300', '3': '301', '4': '302',
    '5': '310', '6': '311', '7': '312',
    '8': '320', '9': '321', 'A': '322',

    'B': '400', 'C': '401', 'D': '402',
    'E': '410', 'F': '411', 'G': '412',
    'H': '420', 'I': '421', 'J': '422',
}

COLOR_DICT = {
    '.': [0xFF, 0xFF, 0xFF],
    'r': [0xFF, 0x00, 0x00],
    'y': [0xFF, 0xFF, 0x00],
    'g': [0x00, 0xFF, 0x00],
    'w': [0xD3, 0xD3, 0xD3],
    'o': [0xFF, 0xA5, 0x00],
    'b': [0x00, 0x00, 0xFF],
    'u': [0x00, 0x00, 0x00]
}

map_concept = '''
outline = 2
M(7+9k, 9+12k) = 
    ....................
    ....................
    ..j..............D..
    ..mk.....a......CG..
    ..pnl...d.b....BFJ..
    ...qo..g.e.c...EI...
    ....r.s.h.f.4..H....
    ......vt.i.37.......
    ......ywu.26A.......
    .......zx.59........
    ........1.8.........
    ....................
    ....................
'''
MAP = [
    '.........................................................................................................',
    '.........................................................................................................',
    '.........................................................................................................',
    '...j.................................................................................................D...',
    '...jj...............................................a...............................................DD...',
    '...jjj.............................................aaa.............................................DDD...',
    '...jjjj...........................................aaaaa...........................................DDDD...',
    '...jjjjj.........................................aaaaaaa.........................................DDDDD...',
    '...jjjjjj.......................................aaaaaaaaa.......................................DDDDDD...',
    '...jjjjjjj.....................................aaaaaaaaaaa.....................................DDDDDDD...',
    '...jjjjjjj....................................aaaaaaaaaaaaa....................................DDDDDDD...',
    '....jjjjjj.k.................................aaaaaaaaaaaaaaa.................................C.DDDDDD....',
    '...m.jjjjj.kk...............................d.aaaaaaaaaaaaa.b...............................CC.DDDDD.G...',
    '...mm.jjjj.kkk.............................ddd.aaaaaaaaaaa.bbb.............................CCC.DDDD.GG...',
    '...mmm.jjj.kkkk...........................ddddd.aaaaaaaaa.bbbbb...........................CCCC.DDD.GGG...',
    '...mmmm.jj.kkkkk.........................ddddddd.aaaaaaa.bbbbbbb.........................CCCCC.DD.GGGG...',
    '...mmmmm.j.kkkkkk.......................ddddddddd.aaaaa.bbbbbbbbb.......................CCCCCC.D.GGGGG...',
    '...mmmmmm..kkkkkkk.....................ddddddddddd.aaa.bbbbbbbbbbb.....................CCCCCCC..GGGGGG...',
    '...mmmmmmm.kkkkkkk....................ddddddddddddd.a.bbbbbbbbbbbbb....................CCCCCCC.GGGGGGG...',
    '...mmmmmmm..kkkkkk.l.................ddddddddddddddd.bbbbbbbbbbbbbbb.................B.CCCCCC..GGGGGGG...',
    '....mmmmmm.n.kkkkk.ll...............g.ddddddddddddd.e.bbbbbbbbbbbbb.c...............BB.CCCCC.F.GGGGGG....',
    '...p.mmmmm.nn.kkkk.lll.............ggg.ddddddddddd.eee.bbbbbbbbbbb.ccc.............BBB.CCCC.FF.GGGGG.J...',
    '...pp.mmmm.nnn.kkk.llll...........ggggg.ddddddddd.eeeee.bbbbbbbbb.ccccc...........BBBB.CCC.FFF.GGGG.JJ...',
    '...ppp.mmm.nnnn.kk.lllll.........ggggggg.ddddddd.eeeeeee.bbbbbbb.ccccccc.........BBBBB.CC.FFFF.GGG.JJJ...',
    '...pppp.mm.nnnnn.k.llllll.......ggggggggg.ddddd.eeeeeeeee.bbbbb.ccccccccc.......BBBBBB.C.FFFFF.GG.JJJJ...',
    '...ppppp.m.nnnnnn..lllllll.....ggggggggggg.ddd.eeeeeeeeeee.bbb.ccccccccccc.....BBBBBBB..FFFFFF.G.JJJJJ...',
    '...pppppp..nnnnnnn.lllllll....ggggggggggggg.d.eeeeeeeeeeeee.b.ccccccccccccc....BBBBBBB.FFFFFFF..JJJJJJ...',
    '...ppppppp.nnnnnnn..llllll...ggggggggggggggg.eeeeeeeeeeeeeee.ccccccccccccccc...BBBBBB..FFFFFFF.JJJJJJJ...',
    '...ppppppp..nnnnnn.o.lllll....ggggggggggggg.h.eeeeeeeeeeeee.f.ccccccccccccc....BBBBB.E.FFFFFF..JJJJJJJ...',
    '....pppppp.q.nnnnn.oo.llll...s.ggggggggggg.hhh.eeeeeeeeeee.fff.ccccccccccc.4...BBBB.EE.FFFFF.I.JJJJJJ....',
    '.....ppppp.qq.nnnn.ooo.lll...ss.ggggggggg.hhhhh.eeeeeeeee.fffff.ccccccccc.44...BBB.EEE.FFFF.II.JJJJJ.....',
    '......pppp.qqq.nnn.oooo.ll...sss.ggggggg.hhhhhhh.eeeeeee.fffffff.ccccccc.444...BB.EEEE.FFF.III.JJJJ......',
    '.......ppp.qqqq.nn.ooooo.l...ssss.ggggg.hhhhhhhhh.eeeee.fffffffff.ccccc.4444...B.EEEEE.FF.IIII.JJJ.......',
    '........pp.qqqqq.n.oooooo....sssss.ggg.hhhhhhhhhhh.eee.fffffffffff.ccc.44444....EEEEEE.F.IIIII.JJ........',
    '.........p.qqqqqq..ooooooo...ssssss.g.hhhhhhhhhhhhh.e.fffffffffffff.c.444444...EEEEEEE..IIIIII.J.........',
    '...........qqqqqqq.ooooooo...sssssss.hhhhhhhhhhhhhhh.fffffffffffffff.4444444...EEEEEEE.IIIIIII...........',
    '...........qqqqqqq..oooooo...sssssss..hhhhhhhhhhhhh.i.fffffffffffff..4444444...EEEEEE..IIIIIII...........',
    '............qqqqqq.r.ooooo....ssssss.t.hhhhhhhhhhh.iii.fffffffffff.3.444444....EEEEE.H.IIIIII............',
    '.............qqqqq.rr.oooo...v.sssss.tt.hhhhhhhhh.iiiii.fffffffff.33.44444.7...EEEE.HH.IIIII.............',
    '..............qqqq.rrr.ooo...vv.ssss.ttt.hhhhhhh.iiiiiii.fffffff.333.4444.77...EEE.HHH.IIII..............',
    '...............qqq.rrrr.oo...vvv.sss.tttt.hhhhh.iiiiiiiii.fffff.3333.444.777...EE.HHHH.III...............',
    '................qq.rrrrr.o...vvvv.ss.ttttt.hhh.iiiiiiiiiii.fff.33333.44.7777...E.HHHHH.II................',
    '.................q.rrrrrr....vvvvv.s.tttttt.h.iiiiiiiiiiiii.f.333333.4.77777....HHHHHH.I.................',
    '...................rrrrrrr...vvvvvv..ttttttt.iiiiiiiiiiiiiii.3333333..777777...HHHHHHH...................',
    '...................rrrrrrr...vvvvvvv.ttttttt..iiiiiiiiiiiii..3333333.7777777...HHHHHHH...................',
    '....................rrrrrr...vvvvvvv..tttttt.u.iiiiiiiiiii.2.333333..7777777...HHHHHH....................',
    '.....................rrrrr....vvvvvv.w.ttttt.uu.iiiiiiiii.22.33333.6.777777....HHHHH.....................',
    '......................rrrr...y.vvvvv.ww.tttt.uuu.iiiiiii.222.3333.66.77777.A...HHHH......................',
    '.......................rrr...yy.vvvv.www.ttt.uuuu.iiiii.2222.333.666.7777.AA...HHH.......................',
    '........................rr...yyy.vvv.wwww.tt.uuuuu.iii.22222.33.6666.777.AAA...HH........................',
    '.........................r...yyyy.vv.wwwww.t.uuuuuu.i.222222.3.66666.77.AAAA...H.........................',
    '.............................yyyyy.v.wwwwww..uuuuuuu.2222222..666666.7.AAAAA.............................',
    '.............................yyyyyy..wwwwwww.uuuuuuu.2222222.6666666..AAAAAA.............................',
    '.............................yyyyyyy.wwwwwww..uuuuuu.222222..6666666.AAAAAAA.............................',
    '.............................yyyyyyy..wwwwww.x.uuuuu.22222.5.666666..AAAAAAA.............................',
    '..............................yyyyyy.z.wwwww.xx.uuuu.2222.55.66666.9.AAAAAA..............................',
    '...............................yyyyy.zz.wwww.xxx.uuu.222.555.6666.99.AAAAA...............................',
    '................................yyyy.zzz.www.xxxx.uu.22.5555.666.999.AAAA................................',
    '.................................yyy.zzzz.ww.xxxxx.u.2.55555.66.9999.AAA.................................',
    '..................................yy.zzzzz.w.xxxxxx...555555.6.99999.AA..................................',
    '...................................y.zzzzzz..xxxxxxx.5555555..999999.A...................................',
    '.....................................zzzzzzz.xxxxxxx.5555555.9999999.....................................',
    '.....................................zzzzzzz..xxxxxx.555555..9999999.....................................',
    '......................................zzzzzz.1.xxxxx.55555.8.999999......................................',
    '.......................................zzzzz.11.xxxx.5555.88.99999.......................................',
    '........................................zzzz.111.xxx.555.888.9999........................................',
    '.........................................zzz.1111.xx.55.8888.999.........................................',
    '..........................................zz.11111.x.5.88888.99..........................................',
    '...........................................z.111111...888888.9...........................................',
    '.............................................1111111.8888888.............................................',
    '.............................................1111111.8888888.............................................',
    '..............................................111111.888888..............................................',
    '...............................................11111.88888...............................................',
    '................................................1111.8888................................................',
    '.................................................111.888.................................................',
    '..................................................11.88..................................................',
    '...................................................1.8...................................................',
    '.........................................................................................................',
    '.........................................................................................................',
    '.........................................................................................................'
]


def make_map(cuber_map, img_map):
    row = len(img_map)
    col = len(img_map[0])
    map = [['.' for _ in range(col)] for _ in range(row)]
    for r in range(row):
        for c in range(col):
            key = img_map[r][c]
            if key in POS_DICT.keys():
                pos = POS_DICT[key]
                map[r][c] = cuber_map[int(pos[0])][int(pos[1])][int(pos[2])]
    return map


def make_img(cuber_map, img_map=None, save_p=None, plot=False):
    if img_map is None:
        img_map = MAP
        # img_map = cuber_map()
    row = len(img_map)
    col = len(img_map[0])
    map = make_map(cuber_map, img_map)
    img = [[[COLOR_DICT[map[r][c]][ch] for c in range(col)] for r in range(row)] for ch in range(3)]
    r = Image.fromarray(np.array(img[0])).convert('L')
    g = Image.fromarray(np.array(img[1])).convert('L')
    b = Image.fromarray(np.array(img[2])).convert('L')
    image = Image.merge('RGB', (r, g, b))
    image = image.resize((92, 76))
    show_img = image.resize((256, 192))
    if save_p:
        show_img.save(save_p)
    if plot:
        plt.imshow(show_img)
        plt.show()
    np_img = np.array(image, dtype=np.float32)
    np_img = np.transpose(np_img, [2, 0, 1])
    np_img = np_img / 256.0 -  0.5
    return np_img