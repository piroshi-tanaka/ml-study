"""
行動パターン地図可視化モジュール

foliumを使用してEV行動パターンを地図上に可視化します。
"""

import pandas as pd
import folium
from folium import plugins
from typing import Dict, Optional
import os


class BehaviorMapVisualizer:
    """
    行動パターン地図可視化クラス
    
    foliumを使用してEV行動パターンを地図上に可視化します。
    - 日ごとの切り替え（タイムスライダー）
    - セッション詳細ポップアップ
    - GoogleMapsリンク
    """
    
    def __init__(self, lat_col: str = 'start_lat', lon_col: str = 'start_lon',
                 end_lat_col: Optional[str] = 'end_lat', end_lon_col: Optional[str] = 'end_lon'):
        """
        初期化
        
        Args:
            lat_col (str): 開始緯度カラム名
            lon_col (str): 開始経度カラム名
            end_lat_col (Optional[str]): 終了緯度カラム名（moving用）
            end_lon_col (Optional[str]): 終了経度カラム名（moving用）
        """
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.end_lat_col = end_lat_col
        self.end_lon_col = end_lon_col
    
    def create_behavior_map(self,
                           sessions_df: pd.DataFrame,
                           hashvin: str,
                           daily_vectors_with_clusters: Optional[pd.DataFrame] = None,
                           output_path: str = 'behavior_map.html',
                           zoom_start: int = 12) -> folium.Map:
        """
        hashvinの行動パターン地図を作成
        
        Args:
            sessions_df (pd.DataFrame): セッションデータ（ev_sessions_0.csv相当）
            hashvin (str): 対象のhashvin
            daily_vectors_with_clusters (Optional[pd.DataFrame]): クラスタID付き日次ベクトル
            output_path (str): 出力HTMLパス
            zoom_start (int): 初期ズームレベル
            
        Returns:
            folium.Map: 生成された地図オブジェクト
        """
        print(f"\n{'='*60}")
        print(f"地図作成: {hashvin}")
        print(f"{'='*60}")
        
        # 対象hashvinのデータ抽出
        hv_sessions = sessions_df[sessions_df['hashvin'] == hashvin].copy()
        
        if len(hv_sessions) == 0:
            print(f"警告: {hashvin}のセッションが見つかりません")
            # 空の地図を返す
            m = folium.Map(location=[35.6812, 139.7671], zoom_start=10)
            m.save(output_path)
            return m
        
        # 時刻をdatetimeに変換（format='mixed'で混在形式に対応）
        hv_sessions['start_time'] = pd.to_datetime(hv_sessions['start_time'], format='mixed')
        hv_sessions['end_time'] = pd.to_datetime(hv_sessions['end_time'], format='mixed')
        hv_sessions = hv_sessions.sort_values('start_time')
        
        # 中心座標を計算
        center_lat = hv_sessions[self.lat_col].mean()
        center_lon = hv_sessions[self.lon_col].mean()
        
        # 地図初期化
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # TimestampedGeoJson用のフィーチャーリストを作成
        features = []
        session_num = 0
        
        for idx, row in hv_sessions.iterrows():
            session_num += 1
            
            # 日付を取得（06:00起点の日付）
            date_key = self._get_date_key(row['start_time'])
            date_str = date_key  # ISO8601形式
            
            # クラスタIDを取得
            cluster_id = self._get_cluster_id(row, daily_vectors_with_clusters)
            
            # セッションタイプに応じてFeatureを追加
            if row['session_type'] in ['inactive', 'charging']:
                feature = self._create_marker_feature(
                    row, session_num, date_str, cluster_id
                )
                if feature:
                    features.append(feature)
            
            elif row['session_type'] == 'moving' and self.end_lat_col and self.end_lon_col:
                # movingセッション用のポリライン
                feature = self._create_polyline_feature(
                    row, session_num, date_str
                )
                if feature:
                    features.append(feature)
        
        # TimestampedGeoJsonプラグインを追加
        # add_last_point=Falseにすることで、前日のデータは消える
        plugins.TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period='P1D',  # 1日ごと
            add_last_point=False,  # 前日データを消す
            auto_play=False,
            loop=False,
            max_speed=1,
            loop_button=True,
            date_options='YYYY-MM-DD',
            time_slider_drag_update=True,
            duration='P1D'  # 表示期間を1日に設定
        ).add_to(m)
        
        # 凡例を追加
        self._add_legend(m)
        
        # HTMLとして保存
        m.save(output_path)
        print(f"地図を保存しました: {output_path}")
        print(f"  セッション数: {len(hv_sessions)}")
        
        return m
    
    def _get_date_key(self, dt: pd.Timestamp) -> str:
        """06:00起点の日付キーを取得（ISO8601形式）"""
        if dt.hour < 6:
            date = (dt - pd.Timedelta(days=1)).date()
        else:
            date = dt.date()
        return date.strftime('%Y-%m-%d')
    
    def _get_cluster_id(self, row: pd.Series, 
                       daily_vectors: Optional[pd.DataFrame]) -> str:
        """セッションのクラスタIDを取得"""
        # session_clusterカラムがあればそれを使用
        if 'session_cluster' in row.index:
            return str(row['session_cluster'])
        
        # なければdaily_vectorsから取得を試みる
        if daily_vectors is not None:
            try:
                hashvin = row['hashvin']
                date_key = self._get_date_key(row['start_time'])
                date06 = pd.to_datetime(date_key + ' 06:00:00')
                
                if isinstance(daily_vectors.index, pd.MultiIndex):
                    day_pattern = daily_vectors.loc[(hashvin, date06), 'day_pattern_cluster']
                else:
                    mask = (daily_vectors['hashvin'] == hashvin) & \
                           (daily_vectors['date06'] == date06)
                    day_pattern = daily_vectors.loc[mask, 'day_pattern_cluster'].iloc[0]
                
                return f"P{int(day_pattern)}" if day_pattern >= 0 else "N/A"
            except Exception:
                pass
        
        return "N/A"
    
    def _create_marker_feature(self, 
                               row: pd.Series,
                               session_num: int,
                               date_str: str,
                               cluster_id: str) -> Dict:
        """
        マーカー用GeoJSON Featureを作成
        
        Args:
            row (pd.Series): セッション行
            session_num (int): セッション番号
            date_str (str): 日付文字列
            cluster_id (str): クラスタラベル
            
        Returns:
            Dict: GeoJSON Feature
        """
        lat = row[self.lat_col]
        lon = row[self.lon_col]
        
        # マーカーの色
        if row['session_type'] == 'charging':
            color = 'red'
            type_label = '⚡充電'
        else:  # inactive
            color = 'blue'
            type_label = '🅿️放置'
        
        # ポップアップ内容
        popup_html = self._create_popup_html(row, session_num, cluster_id, type_label)
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            },
            'properties': {
                'time': date_str,
                'popup': popup_html,
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'color': 'white',
                    'weight': 2,
                    'radius': 8
                },
                'style': {'weight': 0},
                'id': f'session_{session_num}'
            }
        }
        
        return feature
    
    def _create_polyline_feature(self,
                                 row: pd.Series,
                                 session_num: int,
                                 date_str: str) -> Optional[Dict]:
        """
        移動セッション用のポリラインFeatureを作成
        
        Args:
            row (pd.Series): セッション行
            session_num (int): セッション番号
            date_str (str): 日付文字列
            
        Returns:
            Optional[Dict]: GeoJSON Feature
        """
        if pd.isna(row.get(self.end_lat_col)) or pd.isna(row.get(self.end_lon_col)):
            return None
        
        start_lat = row[self.lat_col]
        start_lon = row[self.lon_col]
        end_lat = row[self.end_lat_col]
        end_lon = row[self.end_lon_col]
        
        # ポップアップHTML
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #2e7d32;">🚗 移動 #{session_num}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>開始:</b></td><td>{row['start_time']}</td></tr>
                <tr><td><b>終了:</b></td><td>{row['end_time']}</td></tr>
                <tr>
                    <td colspan="2" style="padding-top: 5px;">
                        <a href="https://www.google.com/maps/dir/{start_lat},{start_lon}/{end_lat},{end_lon}" 
                           target="_blank" style="color: #4285f4;">📍 ルートを見る</a>
                    </td>
                </tr>
            </table>
        </div>
        """
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[start_lon, start_lat], [end_lon, end_lat]]
            },
            'properties': {
                'time': date_str,
                'popup': popup_html,
                'style': {
                    'color': 'green',
                    'weight': 3,
                    'opacity': 0.7
                },
                'id': f'moving_{session_num}'
            }
        }
        
        return feature
    
    def _create_popup_html(self, row: pd.Series, idx: int, 
                          cluster_id: str, type_label: str) -> str:
        """ポップアップHTMLを作成"""
        lat = row[self.lat_col]
        lon = row[self.lon_col]
        
        # GoogleMapsリンク
        gmaps_url = f"https://www.google.com/maps?q={lat},{lon}"
        
        # SOC情報（あれば）
        soc_info = ""
        if 'start_soc' in row.index and pd.notna(row['start_soc']):
            start_soc = row['start_soc']
            end_soc = row.get('end_soc', start_soc)
            soc_change = end_soc - start_soc
            soc_arrow = "📈" if soc_change > 0 else "📉" if soc_change < 0 else "➡️"
            
            soc_info = f"""
            <tr><td><b>開始SOC:</b></td><td>{start_soc:.1f}%</td></tr>
            <tr><td><b>終了SOC:</b></td><td>{end_soc:.1f}%</td></tr>
            <tr><td><b>SOC変化:</b></td><td>{soc_arrow} {soc_change:+.1f}%</td></tr>
            """
        
        # 滞在時間
        duration = row.get('duration_minutes', 0)
        duration_str = f"{int(duration//60)}時間{int(duration%60)}分"
        
        html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #1976d2;">
                {type_label} #{idx}
            </h4>
            <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                <tr style="background-color: #fff9c4;">
                    <td colspan="2" style="padding: 5px; text-align: center;">
                        <b style="font-size: 14px;">クラスタID: {cluster_id}</b>
                    </td>
                </tr>
                <tr><td style="padding-top: 8px;"><b>開始時刻:</b></td><td style="padding-top: 8px;">{row['start_time']}</td></tr>
                <tr><td><b>終了時刻:</b></td><td>{row['end_time']}</td></tr>
                <tr><td><b>滞在時間:</b></td><td>{duration_str}</td></tr>
                {soc_info}
                <tr><td><b>位置:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                <tr>
                    <td colspan="2" style="padding-top: 8px; text-align: center;">
                        <a href="{gmaps_url}" target="_blank" 
                           style="background-color: #4285f4; color: white; padding: 6px 12px; 
                                  text-decoration: none; border-radius: 4px; display: inline-block;">
                            📍 Google Mapsで開く
                        </a>
                    </td>
                </tr>
            </table>
        </div>
        """
        
        return html
    
    def _add_legend(self, m: folium.Map):
        """凡例を追加"""
        legend_html = '''
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            background-color: white;
            padding: 15px;
            border: 2px solid #ccc;
            border-radius: 8px;
            z-index: 1000;
            font-family: Arial, sans-serif;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            <h4 style="margin: 0 0 10px 0; font-size: 14px; border-bottom: 2px solid #1976d2; padding-bottom: 5px;">
                📊 凡例
            </h4>
            <div style="font-size: 12px; line-height: 1.8;">
                <div>🔵 放置セッション</div>
                <div>🔴 充電セッション</div>
                <div><span style="color: green; font-weight: bold;">━</span> 移動ルート</div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; font-size: 11px; color: #666;">
                    ※クリックで詳細表示<br>
                    ※スライダーで日付切替
                </div>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.elements.Element(legend_html))


def create_all_behavior_maps(sessions_df: pd.DataFrame,
                             daily_vectors_with_clusters: pd.DataFrame,
                             output_dir: str = 'outputs/maps',
                             lat_col: str = 'start_lat',
                             lon_col: str = 'start_lon') -> Dict[str, str]:
    """
    全hashvinの行動パターン地図を一括作成
    
    Args:
        sessions_df (pd.DataFrame): セッションデータ
        daily_vectors_with_clusters (pd.DataFrame): クラスタID付き日次ベクトル
        output_dir (str): 出力ディレクトリ
        lat_col (str): 緯度カラム名
        lon_col (str): 経度カラム名
        
    Returns:
        Dict[str, str]: hashvin -> 出力ファイルパスのマッピング
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = BehaviorMapVisualizer(lat_col=lat_col, lon_col=lon_col)
    
    hashvins = sessions_df['hashvin'].unique()
    map_paths = {}
    
    print(f"\n{'='*60}")
    print(f"全hashvinの地図を一括作成: {len(hashvins)}台")
    print(f"{'='*60}")
    
    for hashvin in hashvins:
        output_path = os.path.join(output_dir, f'behavior_map_{hashvin}.html')
        
        try:
            visualizer.create_behavior_map(
                sessions_df=sessions_df,
                hashvin=hashvin,
                daily_vectors_with_clusters=daily_vectors_with_clusters,
                output_path=output_path
            )
            map_paths[hashvin] = output_path
        except Exception as e:
            print(f"エラー: {hashvin}の地図作成に失敗しました: {e}")
    
    print(f"\n[OK] {len(map_paths)}個の地図を作成しました")
    print(f"出力先: {output_dir}")
    
    return map_paths
