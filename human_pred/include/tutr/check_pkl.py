import pickle

def check_pkl_file(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 打印数据类型和数据集大小
    print(f"Data type: {type(data)}")
    print(f"Number of trajectories: {len(data)}")
    
    # 打印前几个轨迹序列的详细信息
    for i in range(5):  # 打印前5个轨迹序列
        print(f'Trajectory {i}:')
        print('Data:', data[i])
        print()


# 使用示例
check_pkl_file('./dataset/eth_test.pkl')
