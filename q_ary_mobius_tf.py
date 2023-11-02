import pickle
import cmath


def degree(index):
    count = 0
    coordinates = []
    coordinate_values = []
    for i in range(len(index)):
        if index[i] != 0:
            count += 1
            coordinates.append(i)
            coordinate_values.append(index[i])
    return count, coordinates, coordinate_values


def check_if_real(dic):
    sq_sum_real = 0
    sq_sum_imag = 0
    cnt = 0
    for value in dic.values():
        real_part = value.real ** 2
        imag_part = value.imag ** 2
        sq_sum_real += real_part
        sq_sum_imag += imag_part
        cnt += (real_part + imag_part) < 1.0e-6
    print(f"Real_energy: {sq_sum_real}, Imag Energy ={sq_sum_imag}")
    print(f"Total terms={len(dic)}, {cnt} small")
    return sq_sum_real, sq_sum_imag, cnt


def get_second_order_coefficient(q, exponent1, exponent2, value1, value2):
    omega = cmath.exp(2*cmath.pi*1j/q)
    first_order1 = (((omega ** exponent1) ** value1) - 1)
    first_order2 = (((omega ** exponent2) ** value2) - 1)
    curr = omega ** (exponent1*value1 + exponent2*value2)
    return curr - first_order2 - first_order1 - 1


if __name__ == '__main__':
    mobius_tf = dict()
    n = 75
    q = 11
    omega = cmath.exp(2*cmath.pi*1j/q)
    with open('sorted_qsft_transform_2nd_order.pickle', 'rb') as f:
        data = pickle.load(f)
    print(" ")
    print("Fourier Transform")
    print("-------------------------------------------------------")
    check_if_real(data)
    zeros = [0 for _ in range(n)]
    for key, value in zip(data.keys(), data.values()):
        deg, coordinates, coordinate_vals = degree(key)
        if deg == 0:
            mobius_tf[key] = mobius_tf.get(key, 0) + value
        elif deg == 1:
            # Zeroth Order Term
            mobius_tf[tuple(zeros)] = mobius_tf.get(tuple(zeros), 0) + value
            # 1st Order Terms
            coord = coordinates[0]
            for i in range(q-1):
                iter_val = i+1
                exponent = coordinate_vals[0]
                curr_key = zeros.copy()
                curr_key[coord] = iter_val
                mobius_tf[tuple(curr_key)] = mobius_tf.get(tuple(curr_key), 0) + value*(((omega ** exponent) **
                                                                                       iter_val) - 1)
        elif deg == 2:
            # Zeroth Order
            mobius_tf[tuple(zeros)] = mobius_tf.get(tuple(zeros), 0) + value
            # 1st Order Terms
            for coord in coordinates:
                for i in range(q - 1):
                    iter_val = i + 1
                    exponent = coordinate_vals[0]
                    curr_key = zeros.copy()
                    curr_key[coord] = iter_val
                    mobius_tf[tuple(curr_key)] = mobius_tf.get(tuple(curr_key), 0) + value * (((omega ** exponent) **
                                                                                               iter_val) - 1)
            # 2nd Order Terms:
            for i in range(q-1):
                iter_val1 = i+1
                for j in range(q-1):
                    iter_val2 = j+1
                    curr_key = zeros.copy()
                    curr_key[coordinates[0]] = iter_val1
                    curr_key[coordinates[1]] = iter_val2
                    const = get_second_order_coefficient(q, coordinate_vals[0], coordinate_vals[1], iter_val1, iter_val2)
                    mobius_tf[tuple(curr_key)] = mobius_tf.get(tuple(curr_key), 0) + value * const
        else:
            print("ignoring")
    print(" ")
    print("Mobius Transform")
    print("-------------------------------------------------------")
    check_if_real(mobius_tf)
