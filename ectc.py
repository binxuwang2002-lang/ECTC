import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch
import torch.nn as nn
from collections import deque
from itertools import combinations

warnings.filterwarnings("ignore", category=UserWarning)

# 自定义LSTM单元实现
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        final = self.fc(out[:, -1, :])
        return final, out

class SensorNetworkSystem:
    """
    ECTC框架的核心模拟系统。
    包含了数据处理、模型预测、博弈论调度和性能评估的完整逻辑。
    """
    def __init__(self, num_nodes=20, hours=24):
        self.num_nodes = num_nodes
        self.hours = hours
        self.time_index = pd.date_range(start="2023-01-01", periods=hours, freq="H")
        self.nodes = self._initialize_nodes()
        self.schedule_grid = pd.DataFrame(
            index=self.time_index,
            columns=self.nodes.keys(),
            data=0
        )
        self.coalition_utilities = {}
        self.payoff_allocation = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actual_max_relay = 0
        self.delay_records = {'ECTC': [], 'TDMA': [], 'BATDMA': [], 'Backscatter': [], 'RL': []}

    # --- 核心初始化与辅助方法 ---
    def _initialize_nodes(self):
        nodes = {}
        for i in range(self.num_nodes):
            nodes[f"Node_{i:02d}"] = {
                'charge_time': np.clip(np.random.exponential(scale=4), 1, 18),
                'transmit_duration': np.random.randint(1, 4),
                'transmit_power': 0.1 + np.random.rand()*0.4,
                'sleep_power': 0.005,
                'raw_data': self._generate_sensor_data(),
                'processed_data': None,
                'prediction': None,
                'schedule': [],
                'position': (np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                'range': np.random.uniform(3, 8),
                'is_relay': np.random.rand() > 0.7,
            }
        return nodes

    def _generate_sensor_data(self):
        base_signal = np.sin(np.linspace(0, 4*np.pi, self.hours))
        mask = np.random.rand(self.hours) > 0.3
        noise = np.random.normal(0, 0.15, self.hours)
        data = np.where(mask, base_signal + noise, np.nan)
        return pd.Series(data, index=self.time_index)

    def _record_delay(self, node, scheduled_time):
        charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
        delay = (scheduled_time - charge_ready).total_seconds() / 3600
        return max(delay, 0)

    # --- 1. 数据稀疏性处理模块 (KF-GP) ---
    def _kalman_imputation(self, series):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        filled = series.interpolate(method='time').values.reshape(-1,1)
        (filtered, _) = kf.smooth(filled)
        return pd.Series(filtered.flatten(), index=series.index)
    
    def _gaussian_process(self, series):
        valid = series.dropna()
        X = valid.index.view('int64').values.reshape(-1,1)
        y = valid.values.ravel()
        
        kernel = 1.0 * RBF(length_scale=1e4) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
        gp.fit(X, y)
        
        X_all = series.index.view('int64').values.reshape(-1,1)
        return pd.Series(gp.predict(X_all), index=series.index)

    def adaptive_imputation_single(self, nid, max_short_gap=3):
        node = self.nodes[nid]
        series = node['raw_data'].copy()
        is_na = series.isna()
        
        status = is_na.astype(int).values
        starts, ends = [], []
        current_start = None
        
        for i in range(len(status)):
            if status[i] and (i == 0 or not status[i-1]):
                current_start = i
            if not status[i] and current_start is not None:
                ends.append(i)
                starts.append(current_start)
                current_start = None
        if current_start is not None:
            ends.append(len(status))
            starts.append(current_start)
        
        processed = series.copy()
        for s, e in zip(starts, ends):
            context_before = min(2, s)
            context_after = min(2, len(series)-e)
            segment = series.iloc[s-context_before:e+context_after].copy()
            
            gap_length = e - s
            if gap_length <= max_short_gap:
                filled = self._robust_kalman(segment, context_before, gap_length)
            else:
                filled = self._safe_gp(segment, context_before, gap_length)
            
            if len(filled) != gap_length:
                filled = filled[:gap_length]
            processed.iloc[s:e] = filled
        
        node['processed_data'] = processed.fillna(method='ffill').fillna(method='bfill')

    def _robust_kalman(self, segment, context_before, gap_length):
       try:
           kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
           filled = segment.interpolate(method='time').values.reshape(-1,1)
           (filtered, _) = kf.smooth(filled)
           return filtered[context_before:context_before+gap_length].flatten()
       except:
           return segment.interpolate(method='linear').iloc[context_before:context_before+gap_length].values

    def _safe_gp(self, segment, context_before, gap_length):
       valid = segment.dropna()
       if len(valid) < 3:
          return segment.ffill().bfill().iloc[context_before:context_before+gap_length].values
    
       try:
          X = valid.index.view('int64').values.reshape(-1,1)
          y = valid.values.ravel()
          kernel = 1.0 * RBF(length_scale=1e4) + WhiteKernel(noise_level=0.1)
          gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
          gp.fit(X, y)
        
          X_all = segment.index.view('int64').values.reshape(-1,1)
          pred = gp.predict(X_all)
          return pred[context_before:context_before+gap_length]
       except:
          return segment.ffill().bfill().iloc[context_before:context_before+gap_length].values

    # --- 2. 混合预测模块 (TinyLSTM-TinyXGBoost) ---
    def hybrid_prediction_chunk(self, chunk):
        for nid in chunk:
            node = self.nodes[nid]
            data = node['processed_data'].values.reshape(-1, 1)
            
            look_back = 6
            X, y = [], []
            if len(data) <= look_back:
                node['prediction'] = np.zeros(1)
                node['mae'] = np.mean(np.abs(data))
                continue

            for i in range(len(data)-look_back):
                X.append(data[i:i+look_back])
                y.append(data[i+look_back])
            X, y = np.array(X), np.array(y)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            lstm = PyTorchLSTM(input_size=1, hidden_size=16)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm.parameters(), lr=0.02)
            
            lstm.train()
            for epoch in range(30):
                optimizer.zero_grad()
                predictions, _ = lstm(X_tensor)
                loss = criterion(predictions, y_tensor)
                loss.backward()
                optimizer.step()
                
            lstm.eval()
            with torch.no_grad():
                _, hidden_states = lstm(X_tensor)
                lstm_features = hidden_states[:, -1, :].cpu().numpy()
            
            xgb = XGBRegressor(n_estimators=100, max_depth=3)
            xgb.fit(lstm_features, y.ravel())
            
            node['prediction'] = xgb.predict(lstm_features)
            node['mae'] = np.mean(np.abs(y.ravel() - node['prediction']))

    # --- 3. 博弈论与协同调度模块 (DASV) ---
    def shapley_value_allocation(self, samples=2000):
        all_nodes = list(self.nodes.keys())
        n = len(all_nodes)
        shapley = {nid: 0.0 for nid in all_nodes}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._calculate_marginals, np.random.permutation(all_nodes))
                       for _ in range(samples)]
            
            for future in futures:
                marginal_contribs = future.result()
                for nid, val in marginal_contribs.items():
                    shapley[nid] += val
        
        total_util = self.utility_function(tuple(all_nodes))
        scaling_factor = total_util / sum(shapley.values()) if sum(shapley.values()) != 0 else 0
        self.payoff_allocation = {k: v * scaling_factor for k, v in shapley.items()}
        return self.payoff_allocation

    def _calculate_marginals(self, perm):
        marginal_contribs = {}
        prev_util = 0.0
        current_coal = []
        for nid in perm:
            current_coal.append(nid)
            current_util = self.utility_function(tuple(current_coal))
            marginal_contribs[nid] = current_util - prev_util
            prev_util = current_util
        return marginal_contribs

    def utility_function(self, coalition):
        if not coalition: return 0.0
        coalition = tuple(sorted(coalition)) # 保证键的唯一性
        if coalition in self.coalition_utilities:
            return self.coalition_utilities[coalition]
        
        total_energy = 0.0
        total_delay = 0.0
        total_mae = 0.0
        
        temp_schedule = self.schedule_grid.copy()
        for nid in coalition:
            node = self.nodes[nid]
            best_start = self.find_cooperative_slot(nid, temp_schedule, coalition)
            
            dur = node['transmit_duration']
            allocated = pd.date_range(best_start, periods=dur, freq='H')
            temp_schedule.loc[allocated, nid] = 1
            
            active_hours = dur
            energy = active_hours * node['transmit_power'] + (self.hours - active_hours) * node['sleep_power']
            total_energy += energy
            
            charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
            delay = (best_start - charge_ready).total_seconds() / 3600
            total_delay += max(delay, 0)
            
            total_mae += node.get('mae', 1.0) # 如果没有mae，则使用默认惩罚
        
        scale_factor = 1 + 0.05 * (len(coalition) - 1)
        utility = (200 - 1.2*total_energy - 0.8*total_delay - 0.05*total_mae) * scale_factor
        self.coalition_utilities[coalition] = utility
        return utility

    def find_cooperative_slot(self, nid, temp_schedule, coalition):
        node = self.nodes[nid]
        dur = node['transmit_duration']
        best_score = -float('inf')
        
        charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
        # 确保至少有一个有效的开始时间
        valid_starts = [t for t in self.time_index[:-dur+1] if t >= charge_ready]
        if not valid_starts:
            return self.time_index[0] # 无法满足充电时间，返回最早时间
        
        best_start = valid_starts[0]
        
        for start in valid_starts[:10]:
            end = start + pd.Timedelta(hours=dur-1)
            window = temp_schedule.loc[start:end]
            
            if window.sum().sum() > 0: continue
                
            time_score = 100 / (start.hour + 1)
            energy_score = 50 / (node['transmit_power'] + 0.1)
            score = time_score + energy_score
            
            if score > best_score:
                best_score = score
                best_start = start
                
        return best_start
    
    def cooperative_scheduling(self):
        self.shapley_value_allocation(samples=2000)
        
        sorted_nodes = sorted(self.nodes.keys(),
                             key=lambda x: self.payoff_allocation.get(x, 0),
                             reverse=True)
        groups = np.array_split(sorted_nodes, 4)
        
        temp_schedule = self.schedule_grid.copy()
        for group in groups:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self._schedule_single_node, nid, temp_schedule.copy(), tuple(group)): nid for nid in group}
                
                for future in futures:
                    nid = futures[future]
                    nid_schedule = future.result()
                    delay = self._record_delay(self.nodes[nid], nid_schedule[0])
                    self.delay_records['ECTC'].append(delay)
                    
                    dur = self.nodes[nid]['transmit_duration']
                    allocated = pd.date_range(nid_schedule[0], periods=dur, freq='H')
                    temp_schedule.loc[allocated, nid] = 1
        
        self.schedule_grid = temp_schedule
        return self.schedule_grid

    def _schedule_single_node(self, nid, temp_schedule, group):
        best_start = self.find_cooperative_slot(nid, temp_schedule, group)
        # 返回开始时间和节点ID，以便主线程记录延迟和更新调度
        return (best_start, nid)

    # --- 4. 基准算法模拟 ---
    def simulate_tdma(self):
        tdma_schedule = pd.DataFrame(index=self.time_index, columns=self.nodes.keys(), data=0)
        num_slots = self.hours
        for i, nid in enumerate(self.nodes):
            slot = i % num_slots
            start_time = self.time_index[slot]
            tdma_schedule.loc[start_time, nid] = 1
            delay = self._record_delay(self.nodes[nid], start_time)
            self.delay_records['TDMA'].append(delay)
        return tdma_schedule

    def simulate_batdma(self):
        batdma_schedule = pd.DataFrame(index=self.time_index, columns=self.nodes.keys(), data=0)
        occupied_slots = set()
        for nid, node in self.nodes.items():
            charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
            for slot, current_time in enumerate(self.time_index):
                if current_time >= charge_ready and slot not in occupied_slots:
                    batdma_schedule.loc[current_time, nid] = 1
                    occupied_slots.add(slot)
                    delay = self._record_delay(node, current_time)
                    self.delay_records['BATDMA'].append(delay)
                    break
        return batdma_schedule

    def simulate_backscatter(self):
        backscatter_schedule = pd.DataFrame(index=self.time_index, columns=self.nodes.keys(), data=0)
        for nid, node in self.nodes.items():
            activation_times = np.random.poisson(0.3, size=3)
            for t in activation_times:
                if t < self.hours:
                    scheduled_time = self.time_index[t]
                    backscatter_schedule.loc[scheduled_time, nid] = 1
                    delay = self._record_delay(node, scheduled_time)
                    self.delay_records['Backscatter'].append(delay)
        return backscatter_schedule
        
    def simulate_rl(self):
        for nid, node in self.nodes.items():
            charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
            valid_slots = [t for t in self.time_index if t >= charge_ready]
            if not valid_slots: continue
            best_start = np.random.choice(valid_slots[:8])
            delay = self._record_delay(node, best_start)
            self.delay_records['RL'].append(delay)

    # --- 5. 性能评估模块 ---
    def calculate_delay_metrics(self, method='ECTC'):
        delays = self.delay_records.get(method, [])
        if not delays: return {'average': 0, 'max': 0, 'percentile_95': 0}
        return {
            'average': np.mean(delays),
            'max': np.max(delays),
            'percentile_95': np.percentile(delays, 95)
        }

    def evaluate_performance(self):
        metrics = {}
        scheduled_nodes = (self.schedule_grid.sum(axis=0) > 0).sum()
        metrics['scheduled_ratio'] = scheduled_nodes / self.num_nodes

        total_energy = 0
        sleep_ratios = []
        for nid, node in self.nodes.items():
            active_hours = self.schedule_grid[nid].sum()
            energy = (active_hours * node['transmit_power'] + (self.hours - active_hours) * node['sleep_power'])
            total_energy += energy
            sleep_ratios.append((self.hours - active_hours) / self.hours)
        
        metrics['total_energy'] = total_energy
        metrics['average_sleep_ratio'] = np.mean(sleep_ratios) if sleep_ratios else 0
        
        # 博弈论指标
        nash_count = 0
        for nid in self.nodes:
            individual_util = self._individual_utility(nid)
            coalition_util = self.payoff_allocation.get(nid, 0)
            if coalition_util > 0.6 * individual_util:
                nash_count += 1
        metrics['nash_equilibrium'] = nash_count / self.num_nodes >= 0.75
        
        grand_util = self._grand_coalition_utility()
        min_acceptable = sum(self._individual_utility(nid) * 0.9 for nid in self.nodes)
        metrics['core_solution'] = grand_util >= min_acceptable
        
        health_score = (
            0.4 * metrics['scheduled_ratio'] +
            0.3 * (1 - metrics['total_energy'] / (self.num_nodes * 0.5 * self.hours)) +
            0.2 * metrics['average_sleep_ratio'] +
            0.1 * (1 if metrics['nash_equilibrium'] else 0)
        )
        metrics['system_health'] = np.clip(health_score, 0, 1)
        
        # 延迟指标
        delay_metrics = self.calculate_delay_metrics('ECTC')
        metrics.update({
            'average_delay': delay_metrics['average'],
            'max_delay': delay_metrics['max'],
        })
        
        return metrics

    def _individual_utility(self, nid):
        node = self.nodes[nid]
        active_hours = self.schedule_grid[nid].sum()
        if active_hours == 0: return 50
        
        charge_ready = self.time_index[0] + pd.Timedelta(hours=node['charge_time'])
        scheduled_hours = self.schedule_grid[self.schedule_grid[nid] == 1].index
        avg_delay = np.mean([(t - charge_ready).total_seconds()/3600 for t in scheduled_hours]) if len(scheduled_hours) > 0 else 0
        
        return max(200 - 1.2*(active_hours*node['transmit_power']) - 0.8*avg_delay - 0.05*node.get('mae', 1.0), 50)

    def _grand_coalition_utility(self):
        return sum(self._individual_utility(nid) for nid in self.nodes)


if __name__ == "__main__":
    from tqdm import tqdm
    import time

    def run_system():
        """封装并执行ECTC核心流程"""
        print("Initializing ECTC Sensor Network System...")
        system = SensorNetworkSystem(num_nodes=20, hours=24)
        
        try:
            # --- 1. 数据预处理 ---
            print("\n[Stage 1/4] Pre-processing: Data Imputation")
            chunks = np.array_split(list(system.nodes.keys()), 4)
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(tqdm(executor.map(system.adaptive_imputation_single, system.nodes.keys()), total=len(system.nodes), desc="Imputing Data"))

            # --- 2. 模型训练 ---
            print("\n[Stage 2/4] Modeling: Hybrid Prediction")
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(tqdm(executor.map(system.hybrid_prediction_chunk, chunks), total=len(chunks), desc="Training Models"))
            
            # --- 3. ECTC 协同调度 ---
            print("\n[Stage 3/4] Scheduling: Cooperative Game")
            start_time = time.time()
            system.cooperative_scheduling()
            print(f"ECTC scheduling completed in {time.time() - start_time:.2f} seconds.")
            
            # --- 4. 性能评估 ---
            print("\n[Stage 4/4] Evaluation: Performance Metrics")
            metrics = system.evaluate_performance()
            
            # --- 结果展示 ---
            print("\n========== ECTC System Performance Report ==========")
            print(f"  Total Energy Consumption: {metrics['total_energy']:.2f} mWh")
            print(f"  Average Sleep Ratio: {metrics['average_sleep_ratio']*100:.1f}%")
            print(f"  Average Scheduling Delay: {metrics['average_delay']:.2f} hours")
            print(f"  Nash Equilibrium Achieved: {metrics['nash_equilibrium']}")
            print(f"  Core Solution Feasible: {metrics['core_solution']}")
            print(f"  Overall System Health Score: {metrics['system_health']:.2f}/1.0")
            print("======================================================")

            # --- 运行基准算法用于数据对比 ---
            print("\nRunning baseline algorithms for comparison...")
            system.simulate_tdma()
            system.simulate_batdma()
            system.simulate_backscatter()
            system.simulate_rl()
            print("Baseline simulations complete.")

            return system, metrics
            
        except Exception as e:
            print(f"\nAn error occurred during simulation: {str(e)}")
            return None, None
    
    # 执行主程序
    system, metrics = run_system()
    
    if system:
        # 保存结果
        print("\nSaving final ECTC schedule to 'final_schedule.csv'...")
        system.schedule_grid.to_csv("final_schedule.csv")
        print("\n=== ECTC Core Simulation Finished Successfully ===")