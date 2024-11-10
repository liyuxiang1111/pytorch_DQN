'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
'''

class Schedule(object):
    def value(self, t):
        """在时间 t 上获得该调度的值"""
        raise NotImplementedError()

class ConstantSchedule(object):
    def __init__(self, value):
        """
        常数调度：值在整个时间中保持不变。

        参数
        ----------
        value: float
            调度的常数值
        """
        self._v = value

    def value(self, t):
        """返回常数值"""
        return self._v

def linear_interpolation(l, r, alpha):
    """线性插值函数，返回 l 和 r 之间的插值，alpha 表示插值比例"""
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """
        分段调度：在每个区间内值可以插值改变。

        参数
        ----------
        endpoints: [(int, int)]
            包含 `(time, value)` 的列表，表示在时间 `t == time` 时返回的值为 `value`。
            所有时间点必须按升序排列。如果 `t` 在两个时间点之间（例如 `(time_a, value_a)`
            和 `(time_b, value_b)`），那么调度返回 `interpolation(value_a, value_b, alpha)`，
            其中 `alpha` 是时间 `t` 在 `time_a` 和 `time_b` 之间的比例。
        interpolation: lambda float, float, float: float
            插值函数，用于计算 t 左右两端值之间的插值。`alpha` 是 t 相对于两个端点的比例。
        outside_value: float
            当 `t` 超出所有指定区间时返回的值。如果为 None，则在请求超出值时会引发 AssertionError。
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """获取时间 t 上的调度值"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                # 计算 t 在区间 (l_t, r_t) 中的比例
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # 如果 t 不在任何分段内，则返回外部值
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        线性调度：在 `schedule_timesteps` 时间步内从 `initial_p` 线性递减到 `final_p`。
        超过该时间步后将返回 `final_p`。

        参数
        ----------
        schedule_timesteps: int
            将 `initial_p` 线性递减到 `final_p` 的时间步数
        initial_p: float
            初始输出值
        final_p: float
            最终输出值
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """获取时间 t 上的调度值"""
        # 计算 t 在 `schedule_timesteps` 内的比例
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
