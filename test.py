def sort_versions(versions):
    """
    对版本号列表进行排序，由大到小（最新版本在前）
    支持格式如：1.0.1, 1.10.1, 2.0.1, 2.0.1-beta, 2.0.1-alpha, 2.11
    规则：
      - 数字部分按主版本、次版本、修订号依次比较
      - 缺少的段位补0（如 2.11 视为 2.11.0）
      - 预发布版本（alpha < beta < gamma）小于同数字部分的正式版
      - 正式版优先级最高，gamma 次之，beta 再次，alpha 最低
    """
    def parse_version(ver):
        ver = ver.strip()
        # 分离预发布后缀（alpha/beta/gamma）
        parts = ver.split('-', 1)
        num_part = parts[0]
        suffix = parts[1] if len(parts) == 2 else ''
        
        # 解析数字部分，最多取三段，不足补0
        nums = num_part.split('.')
        major = int(nums[0]) if len(nums) > 0 else 0
        minor = int(nums[1]) if len(nums) > 1 else 0
        patch = int(nums[2]) if len(nums) > 2 else 0
        
        # 预发布等级映射：alpha=0, beta=1, gamma=2, 正式版=3
        if suffix == '':
            pre_rank = 3
        else:
            suf_low = suffix.lower()
            if suf_low == 'alpha':
                pre_rank = 0
            elif suf_low == 'beta':
                pre_rank = 1
            elif suf_low == 'gamma':
                pre_rank = 2
            else:   # 未知后缀按最低预发布处理
                pre_rank = 0
        
        return (major, minor, patch, pre_rank)
    
    # 使用自定义键进行降序排序
    return sorted(versions, key=parse_version, reverse=True)


# 示例测试
if __name__ == "__main__":
    input_versions = ["1.0.1", "1.10.1", "2.0.1", "2.0.1-beta", "2.0.1-alpha", "2.11"]
    sorted_versions = sort_versions(input_versions)
    print("排序结果:", sorted_versions)
    # 预期输出: ['2.11', '2.0.1', '2.0.1-beta', '2.0.1-alpha', '1.10.1', '1.0.1']