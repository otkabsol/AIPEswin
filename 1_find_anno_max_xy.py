import json


def max_keypoint_coordinates(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 初始化最大坐标值
    max_x = -float('inf')
    max_y = -float('inf')

    # 遍历每个标注
    for annotation in data['annotations']:
        keypoints = annotation['keypoints']

        # 检查长度是否为51（17个关键点 * 3个数值/关键点）
        if len(keypoints) % 3 != 0:
            print("Warning: keypoints length is not a multiple of 3")
            continue

        # 遍历每个关键点
        for i in range(0, len(keypoints), 3):
            x, y = keypoints[i], keypoints[i + 1]
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    return max_x, max_y


# 使用函数
json_file = 'J:\【毕业论文】【AIPE】实验数据\【24 01 13】swin-l\AIPE-swinL\person_keypoints_val2017.json'
max_x, max_y = max_keypoint_coordinates(json_file)
print(f'Max x coordinate: {max_x}')
print(f'Max y coordinate: {max_y}')
