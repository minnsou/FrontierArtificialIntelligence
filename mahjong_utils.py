from collections import defaultdict
import itertools
import numpy as np

def is_valid(seq, NUM_SAME_TILE=4): # 生成された組み合わせが手牌として妥当かどうかを判断する関数　tuple(seq)の一つ一つが一つの状態(手牌)に対応している
    counts = defaultdict(lambda: 0)
    for i in range(0, len(seq)):
        if i + 1 < len(seq) and seq[i] > seq[i + 1]: # 前半の条件はiが一番最後以外は常に成立、後半の条件は昇順に整列するための条件
            return False
        counts[seq[i]] += 1
        if (counts[seq[i]] > NUM_SAME_TILE): return False # 牌の上限枚数を超えたらFalse
    return True
    
def generate_all_l(kind_tile, num_hand, NUM_SAME_TILE=4): # 全ての手牌の組み合わせをタプルで出力する関数
    gen_list = []
    for seq in itertools.product(range(kind_tile), repeat=num_hand):
        if is_valid(seq, NUM_SAME_TILE):
            gen_list.append(seq)
    return gen_list

def state2hist(state, kind_tile): # 手牌(state)を、牌種ごとの枚数のリスト(長さkind_tile)に変換する関数
    hist = [0] * kind_tile # hist = [0,0,...,0]
    for tile in state:
        hist[tile] += 1
    return hist

def states2hists(state_list, kind_tile): # 手牌(state)のリストを、牌種ごとの枚数のリストに変換する関数
    hist_list = []
    for state in state_list:
        #print(state)
        hist = [0] * kind_tile # hist = [0,0,...,0]
        for tile in state:
            hist[tile] += 1
        hist_list.append(hist)
    return hist_list

def state2hist_for_win(state, NUM_KIND_TILES=34): # あがり判定のために牌種の間には0を入れたhistを生成する関数
    hist = [0] * (NUM_KIND_TILES + 9) # hist = [0,0,...,0]
    for tile in state:
        if tile <= 8: # 萬子
            hist[tile] += 1
        elif tile <= 17: # 筒子
            hist[tile + 1] += 1
        elif tile <= 26: # 索子
            hist[tile + 2] += 1
        else: # 字牌
            hist[30 + (tile - 27) * 2] += 1
    return hist

def win_split_sub(hist, two, three, split_state, agari_list):
    if any(x < 0 for x in hist):
        return
    if two == 0 and three == 0:
        agari_list.append(tuple(split_state))
        return
    i = next(i for i, x in enumerate(hist) if x > 0) # histの中でx>０を満たす最小のindexを持ってくる
    next_hist = [x - 2 if i == j else x for j, x in enumerate(hist)]
    if two > 0 and hist[i] == 2: # 雀頭
        win_split_sub(next_hist, two - 1, three, split_state + [(i, i)], agari_list)
    next_hist = [x - 3 if i == j else x for j, x in enumerate(hist)]
    if three > 0 and hist[i] == 3: # 刻子
        win_split_sub(next_hist, two, three - 1, split_state + [(i, i, i)], agari_list)
    next_hist = [x -1 if i <= j <= i + 2 else x for j, x in enumerate(hist)]
    if three > 0 and i + 2 < len(hist): # 順子
        win_split_sub(next_hist, two, three - 1, split_state + [(i, i+1, i+2)], agari_list)
    return 
    
def win_split_main(hist): # あがり判定
    n_two = 1 if sum(hist) % 3 == 2 else 0
    n_three = sum(hist) // 3
    agari_list = []
    win_split_sub(hist, n_two, n_three, [], agari_list)
    if len(agari_list) == 0:
        return (False, set())
    else:
        return (True, agari_list)

def is_tanyao(state):
    for hai in state:
        if hai == 0 or hai == 8:
            return False
    return True

def is_chanta(split_state):
    state_value = True
    for block in split_state:
        if 0 in block or 8 in block:
            continue
        else:
            state_value = False
            break
    return state_value

def is_toitoi(split_state):
    state_value = True
    for block in split_state:
        if len(block) == 2: # 雀頭
            continue
        else:  # 面子
            if block[0] != block[1]:
                state_value = False
                break
    return state_value
    
def is_ipeko(split_state):
    for block in split_state:
        if len(block) == 2:
            continue
        if block[0] != block[1]:
            temp = list(split_state)
            temp.remove(block)
            if block in temp:
                return True
    return False

def hist2onehot(hist, kind_tile, NUM_SAME_TILE=4): # 手牌１つをサイズ(kind_tile, NUM_SAME_TILE)の行列にする
    matrix = np.zeros(shape=(kind_tile, NUM_SAME_TILE))
    for i, num in enumerate(hist):
        if num == 0:
            continue
        else:
            matrix[i][:num] = 1
    return matrix

def hists2onehots(hist_list, kind_tile, NUM_SAME_TILE=4): # 手牌1つ1つをone_hot形式に直す
    onehots = np.zeros((len(hist_list), kind_tile, NUM_SAME_TILE))
    for i in range(len(hist_list)):
        for j, hist_i in enumerate(hist_list[i]):
            if hist_i == 0:
                continue
            else:
                onehots[i][j][:hist_i] = 1
    return onehots
