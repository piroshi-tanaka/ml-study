"""
行動パターンクラスタリングモジュール（Step2）

日次行動ベクトルから、hashvinごとに行動パターンを抽出します。
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from config import BehaviorAnalysisConfig

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class BehaviorPatternClusterer:
    """
    行動パターンクラスタリングクラス（Step2）
    
    日次行動ベクトルから、hashvinごとに行動パターンを抽出します。
    """
    
    def __init__(self, config: BehaviorAnalysisConfig):
        """
        初期化
        
        Args:
            config (BehaviorAnalysisConfig): 設定パラメータ
        """
        self.config = config
        self.clustering_results_: Dict[str, Dict] = {}  # hashvin -> クラスタリング結果
        
    def fit_transform(self, daily_vector_df: pd.DataFrame, 
                     hashvin: str) -> pd.DataFrame:
        """
        特定のhashvinに対してクラスタリングを実行
        
        Args:
            daily_vector_df (pd.DataFrame): 日次行動ベクトル
            hashvin (str): 対象のhashvin
            
        Returns:
            pd.DataFrame: クラスタID付き日次ベクトル
                新カラム: day_pattern_cluster
        """
        print("=" * 60)
        print(f"Step2: 行動パターンクラスタリング (hashvin={hashvin})")
        print("=" * 60)
        
        # 対象hashvinのデータ抽出
        if isinstance(daily_vector_df.index, pd.MultiIndex):
            df = daily_vector_df.loc[hashvin].copy()
        else:
            df = daily_vector_df[daily_vector_df['hashvin'] == hashvin].copy()
        
        # 空日除外
        df_valid = df[df['is_empty_day'] == 0].copy()
        print(f"対象日数: {len(df_valid)}日（空日{len(df) - len(df_valid)}日を除外）")
        
        if len(df_valid) < self.config.k_range[0]:
            print(f"警告: データ数が少なすぎるためクラスタリングをスキップします")
            df['day_pattern_cluster'] = -1
            return df
        
        # 特徴量抽出
        feature_cols = [col for col in df_valid.columns 
                       if col.startswith('ratio__') or col.startswith('trans_') or col == 'unique_clusters_visited']
        X = df_valid[feature_cols].values
        
        # 標準化
        print("\n特徴量の標準化中...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # クラスタリング手法による分岐
        if self.config.clustering_method == 'kmeans':
            labels, clustering_info = self._cluster_kmeans(X_scaled)
        elif self.config.clustering_method == 'hdbscan':
            labels, clustering_info = self._cluster_hdbscan(X_scaled)
        else:
            raise ValueError(f"未対応のクラスタリング手法: {self.config.clustering_method}")
        
        # 結果を格納
        df_valid['day_pattern_cluster'] = labels
        df['day_pattern_cluster'] = -1  # 空日はデフォルト-1
        df.loc[df_valid.index, 'day_pattern_cluster'] = labels
        
        # クラスタリング結果を保存
        self.clustering_results_[hashvin] = {
            'scaler': scaler,
            'feature_cols': feature_cols,
            'labels': labels,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict(),
            'method': self.config.clustering_method,
            **clustering_info
        }
        
        # クラスタサイズを表示
        print("\nクラスタサイズ:")
        for cluster_id, size in sorted(self.clustering_results_[hashvin]['cluster_sizes'].items()):
            cluster_label = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"  {cluster_label}: {size}日")
        
        print(f"\n[OK] クラスタリング完了")
        
        return df
    
    def _cluster_kmeans(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        KMeansによるクラスタリング
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[np.ndarray, Dict]: (ラベル配列, クラスタリング情報)
        """
        print("\n最適クラスタ数を探索中（KMeans）...")
        best_k, metrics_history = self._find_optimal_k(X)
        print(f"  最適k値: {best_k}")
        
        print(f"\nKMeans (k={best_k}) でクラスタリング実行中...")
        kmeans = KMeans(n_clusters=best_k, random_state=self.config.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels, {
            'kmeans': kmeans,
            'best_k': best_k,
            'metrics_history': metrics_history
        }
    
    def _cluster_hdbscan(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        HDBSCANによるクラスタリング
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[np.ndarray, Dict]: (ラベル配列, クラスタリング情報)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCANが利用できません。'pip install hdbscan'でインストールしてください。")
        
        print("\nHDBSCANでクラスタリング実行中...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"  検出クラスタ数: {n_clusters}")
        print(f"  ノイズ点数: {n_noise}")
        
        return labels, {
            'hdbscan': clusterer,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    
    def _find_optimal_k(self, X: np.ndarray) -> Tuple[int, Dict]:
        """
        最適なk値を自動選択
        
        Args:
            X (np.ndarray): 標準化済み特徴量
            
        Returns:
            Tuple[int, Dict]: (最適k値, メトリクス履歴)
        """
        k_min, k_max = self.config.k_range
        k_range = range(k_min, min(k_max + 1, len(X)))
        
        metrics: Dict[str, list] = {
            'k': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # 各指標を計算
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            
            metrics['k'].append(k)
            metrics['silhouette'].append(sil)
            metrics['calinski_harabasz'].append(ch)
            metrics['davies_bouldin'].append(db)
            
            print(f"  k={k}: Silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")
        
        # ランキングベースで最適k選択
        df_metrics = pd.DataFrame(metrics)
        df_metrics['rank_sil'] = df_metrics['silhouette'].rank(ascending=False)
        df_metrics['rank_ch'] = df_metrics['calinski_harabasz'].rank(ascending=False)
        df_metrics['rank_db'] = df_metrics['davies_bouldin'].rank(ascending=True)  # 低いほど良い
        
        # 合成ランク（平均順位が最良のものを選択）
        df_metrics['avg_rank'] = (
            df_metrics['rank_sil'] + df_metrics['rank_ch'] + df_metrics['rank_db']
        ) / 3
        
        best_idx = df_metrics['avg_rank'].idxmin()
        best_k = df_metrics.loc[best_idx, 'k']
        
        return int(best_k), metrics
    
    def get_cluster_profiles(self, daily_vector_df: pd.DataFrame, 
                            hashvin: str) -> Dict[int, pd.DataFrame]:
        """
        各クラスタのプロファイル（平均ベクトル）を取得
        
        Args:
            daily_vector_df (pd.DataFrame): クラスタID付き日次ベクトル
            hashvin (str): 対象のhashvin
            
        Returns:
            Dict[int, pd.DataFrame]: クラスタID -> プロファイルデータフレーム
        """
        if isinstance(daily_vector_df.index, pd.MultiIndex):
            df = daily_vector_df.loc[hashvin].copy()
        else:
            df = daily_vector_df[daily_vector_df['hashvin'] == hashvin].copy()
        
        df_valid = df[df['is_empty_day'] == 0]
        ratio_cols = [col for col in df_valid.columns if col.startswith('ratio__')]
        
        profiles = {}
        for cluster_id in df_valid['day_pattern_cluster'].unique():
            if cluster_id == -1:
                continue
            
            cluster_data = df_valid[df_valid['day_pattern_cluster'] == cluster_id]
            
            # 平均ベクトル
            avg_vector = cluster_data[ratio_cols].mean()
            
            # 曜日分布
            weekday_dist = cluster_data['weekday'].value_counts(normalize=True).sort_index()
            
            # 代表日（平均ベクトルに最も近い日）
            distances = ((cluster_data[ratio_cols] - avg_vector) ** 2).sum(axis=1)
            representative_dates = distances.nsmallest(3).index.tolist()
            
            profiles[cluster_id] = {
                'avg_vector': avg_vector,
                'weekday_dist': weekday_dist,
                'representative_dates': representative_dates,
                'size': len(cluster_data)
            }
        
        return profiles

