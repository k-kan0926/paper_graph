#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert rosbag to CSV for NARX training (p1/p2 based with sensor feedback)

Usage:
  python analyze_bag_withheight_andpressure.py input.bag output.csv
  python analyze_bag_withheight_andpressure.py input.bag output.csv --cmd-scale 0.9/4096

Features:
  - Extracts commanded pressures (p1_cmd, p2_cmd) from /exp topics
  - Extracts measured pressures (p1_meas, p2_meas) from /mpa_pressure
  - Extracts theta from /kinikun1/joint_states
  - Extracts z (absolute) and computes dz (relative) from /arm1/pose
  - Handles missing topics gracefully (fills with NaN)
  - Auto-detects cmd scaling (checks if /mpa_cmd is scaled or in MPa)
"""

import argparse
import os
import numpy as np
import pandas as pd

try:
    import rosbag
except ImportError:
    print("[ERROR] rosbag not found. Install with: pip install rosbag")
    exit(1)


def to_secs(t):
    """Convert ROS time to seconds"""
    return t.to_sec() if hasattr(t, "to_sec") else float(t)


def nearest_interp(t_ref, t_src, y_src):
    """Nearest-neighbor interpolation"""
    if len(t_src) == 0:
        return np.full_like(t_ref, np.nan, dtype=float)
    t_src = np.asarray(t_src)
    y_src = np.asarray(y_src)
    t_ref = np.asarray(t_ref)
    
    idx = np.searchsorted(t_src, t_ref, side="left")
    idx = np.clip(idx, 0, len(t_src) - 1)
    
    # Choose closest neighbor
    mask = (idx > 0) & (
        (idx == len(t_src)) | 
        (np.abs(t_ref - t_src[idx - 1]) <= np.abs(t_ref - t_src[idx]))
    )
    idx[mask] -= 1
    
    return y_src[idx]


def list_topics(bag):
    """Get topic information from bag"""
    info = bag.get_type_and_topic_info().topics
    return {
        t: {"type": meta.msg_type, "count": meta.message_count}
        for t, meta in info.items()
    }


def find_topic(name_hint, topics):
    """Find topic by hint (exact match or substring)"""
    if name_hint in topics:
        return name_hint
    
    # Try exact suffix match
    tails = [t for t in topics if t.endswith(name_hint)]
    if tails:
        return tails[0]
    
    # Try substring match
    parts = [t for t in topics if name_hint in t]
    return parts[0] if parts else None


def pick_joint_topic(bag_path, topics, joint_topic_hint, joint_name, joint_index=None):
    """Find the correct JointState topic"""
    if joint_topic_hint in topics:
        return joint_topic_hint
    
    jt = find_topic(joint_topic_hint, topics)
    if jt:
        return jt
    
    # Search for JointState topics
    js_cands = [
        t for t, meta in topics.items()
        if meta["type"] == "sensor_msgs/JointState" and meta["count"] > 0
    ]
    
    # If joint_name specified, verify it exists
    if joint_name:
        with rosbag.Bag(bag_path, "r") as b:
            for cand in js_cands:
                for topic, msg, stamp in b.read_messages(topics=[cand]):
                    if hasattr(msg, "name") and joint_name in getattr(msg, "name", []):
                        return cand
                    break  # Check only first message
    
    return js_cands[0] if js_cands else None


def main():
    ap = argparse.ArgumentParser(description="Convert rosbag to CSV (p1/p2 based)")
    ap.add_argument("bag", help="Input rosbag file")
    ap.add_argument("csv", help="Output CSV file")
    
    # Topic hints
    ap.add_argument("--p1-cmd-topic", default="/exp/p1_cmd_MPa")
    ap.add_argument("--p2-cmd-topic", default="/exp/p2_cmd_MPa")
    ap.add_argument("--ps-topic", default="/exp/ps_calc_MPa")
    ap.add_argument("--pd-topic", default="/exp/pd_calc_MPa")
    ap.add_argument("--mpa-cmd-topic", default="/mpa_cmd")
    ap.add_argument("--pressure-sensor-topic", default="/mpa_pressure")
    ap.add_argument("--joint-topic", default="/kinikun1/joint_states")
    ap.add_argument("--joint-name", default="arm3_joint")
    ap.add_argument("--joint-index", type=int, default=2)
    ap.add_argument("--height-topic", default="/arm1/pose")
    
    # Scaling
    ap.add_argument("--cmd-scale", default="0.9/4096",
                    help="Scale factor to convert /mpa_cmd back to MPa (default: 0.9/4096)")
    
    # Options
    ap.add_argument("--list", action="store_true", help="List all topics and exit")
    
    args = ap.parse_args()
    
    # Parse scale
    try:
        scale = float(eval(args.cmd_scale, {}, {}))
    except Exception:
        scale = float(args.cmd_scale)
    
    print(f"[INFO] Reading bag: {args.bag}")
    bag = rosbag.Bag(args.bag, "r")
    topics = list_topics(bag)
    
    if args.list:
        print("=== Available topics ===")
        for t, meta in sorted(topics.items()):
            print(f"{t:60s} [{meta['type']}] x{meta['count']}")
        bag.close()
        return
    
    # Resolve topics
    p1_cmd_topic = find_topic(args.p1_cmd_topic, topics)
    p2_cmd_topic = find_topic(args.p2_cmd_topic, topics)
    ps_topic = find_topic(args.ps_topic, topics)
    pd_topic = find_topic(args.pd_topic, topics)
    mpa_cmd_topic = find_topic(args.mpa_cmd_topic, topics)
    pressure_sensor_topic = find_topic(args.pressure_sensor_topic, topics)
    joint_topic = pick_joint_topic(args.bag, topics, args.joint_topic, 
                                     args.joint_name, args.joint_index)
    height_topic = find_topic(args.height_topic, topics)
    
    print("=== Resolved topics ===")
    print(f"p1_cmd_topic         : {p1_cmd_topic}")
    print(f"p2_cmd_topic         : {p2_cmd_topic}")
    print(f"ps_topic             : {ps_topic}")
    print(f"pd_topic             : {pd_topic}")
    print(f"mpa_cmd_topic        : {mpa_cmd_topic}")
    print(f"pressure_sensor_topic: {pressure_sensor_topic}")
    print(f"joint_topic          : {joint_topic}")
    print(f"height_topic         : {height_topic}")
    
    if not joint_topic:
        raise RuntimeError("JointState topic not found. Use --list to check available topics.")
    
    # Data collection
    t_p1_cmd, v_p1_cmd = [], []
    t_p2_cmd, v_p2_cmd = [], []
    t_ps, v_ps = [], []
    t_pd, v_pd = [], []
    t_mpa_cmd, v_mpa_cmd_p1, v_mpa_cmd_p2 = [], [], []
    t_press_sens, v_press_p1, v_press_p2 = [], [], []
    t_joint, v_joint = [], []
    t_height, z_height = [], []
    
    print("[INFO] Reading messages...")
    for topic, msg, stamp in bag.read_messages():
        ts = to_secs(stamp)
        
        # Commanded pressures (MPa)
        if p1_cmd_topic and topic == p1_cmd_topic and hasattr(msg, "data"):
            t_p1_cmd.append(ts)
            v_p1_cmd.append(float(msg.data))
        
        elif p2_cmd_topic and topic == p2_cmd_topic and hasattr(msg, "data"):
            t_p2_cmd.append(ts)
            v_p2_cmd.append(float(msg.data))
        
        # ps/pd (reference)
        elif ps_topic and topic == ps_topic and hasattr(msg, "data"):
            t_ps.append(ts)
            v_ps.append(float(msg.data))
        
        elif pd_topic and topic == pd_topic and hasattr(msg, "data"):
            t_pd.append(ts)
            v_pd.append(float(msg.data))
        
        # /mpa_cmd (scaled values, need to convert back to MPa)
        elif mpa_cmd_topic and topic == mpa_cmd_topic and hasattr(msg, "x") and hasattr(msg, "y"):
            t_mpa_cmd.append(ts)
            v_mpa_cmd_p1.append(float(msg.x) * scale)
            v_mpa_cmd_p2.append(float(msg.y) * scale)
        
        # Measured pressures from sensor
        elif pressure_sensor_topic and topic == pressure_sensor_topic:
            if hasattr(msg, "x") and hasattr(msg, "y"):
                t_press_sens.append(ts)
                v_press_p1.append(float(msg.x))
                v_press_p2.append(float(msg.y))
        
        # Joint angle
        elif joint_topic and topic == joint_topic:
            if hasattr(msg, "position") and hasattr(msg, "name"):
                names = list(msg.name)
                if args.joint_name in names:
                    idx = names.index(args.joint_name)
                else:
                    idx = args.joint_index
                
                if 0 <= idx < len(msg.position):
                    t_joint.append(ts)
                    v_joint.append(float(msg.position[idx]))
        
        # Height (z from pose)
        elif height_topic and topic == height_topic:
            if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
                pos = msg.pose.position
                if hasattr(pos, "z"):
                    t_height.append(ts)
                    z_height.append(float(pos.z))
    
    bag.close()
    
    # Convert to numpy
    t_p1_cmd = np.asarray(t_p1_cmd)
    v_p1_cmd = np.asarray(v_p1_cmd)
    t_p2_cmd = np.asarray(t_p2_cmd)
    v_p2_cmd = np.asarray(v_p2_cmd)
    t_ps = np.asarray(t_ps)
    v_ps = np.asarray(v_ps)
    t_pd = np.asarray(t_pd)
    v_pd = np.asarray(v_pd)
    t_mpa_cmd = np.asarray(t_mpa_cmd)
    v_mpa_cmd_p1 = np.asarray(v_mpa_cmd_p1)
    v_mpa_cmd_p2 = np.asarray(v_mpa_cmd_p2)
    t_press_sens = np.asarray(t_press_sens)
    v_press_p1 = np.asarray(v_press_p1)
    v_press_p2 = np.asarray(v_press_p2)
    t_joint = np.asarray(t_joint)
    v_joint = np.asarray(v_joint)
    t_height = np.asarray(t_height)
    z_height = np.asarray(z_height)
    
    print(f"[INFO] Collected message counts:")
    print(f"  p1_cmd: {len(t_p1_cmd)}, p2_cmd: {len(t_p2_cmd)}")
    print(f"  ps: {len(t_ps)}, pd: {len(t_pd)}")
    print(f"  mpa_cmd: {len(t_mpa_cmd)}")
    print(f"  pressure_sensor: {len(t_press_sens)}")
    print(f"  joint: {len(t_joint)}")
    print(f"  height: {len(t_height)}")
    
    # Choose reference time axis (prefer p1_cmd or ps)
    if len(t_p1_cmd) > 0:
        t_ref = t_p1_cmd
        print(f"[INFO] Using p1_cmd as reference time axis ({len(t_ref)} samples)")
    elif len(t_ps) > 0:
        t_ref = t_ps
        print(f"[INFO] Using ps as reference time axis ({len(t_ref)} samples)")
    else:
        raise RuntimeError("No pressure command data found!")
    
    # Relative time (from first sample)
    t0 = min(t_ref[0], t_joint[0] if len(t_joint) > 0 else t_ref[0])
    if len(t_height) > 0:
        t0 = min(t0, t_height[0])
    ts = t_ref - t0
    
    # Interpolate all signals to reference time
    p1_cmd = v_p1_cmd if len(t_p1_cmd) > 0 else nearest_interp(t_ref, t_p1_cmd, v_p1_cmd)
    p2_cmd = v_p2_cmd if len(t_p2_cmd) > 0 else nearest_interp(t_ref, t_p2_cmd, v_p2_cmd)
    
    ps = nearest_interp(t_ref, t_ps, v_ps) if len(t_ps) > 0 else p1_cmd + p2_cmd
    pd = nearest_interp(t_ref, t_pd, v_pd) if len(t_pd) > 0 else p1_cmd - p2_cmd
    
    # Alternative p1/p2 from /mpa_cmd (if /exp topics missing)
    if len(t_mpa_cmd) > 0:
        p1_cmd_alt = nearest_interp(t_ref, t_mpa_cmd, v_mpa_cmd_p1)
        p2_cmd_alt = nearest_interp(t_ref, t_mpa_cmd, v_mpa_cmd_p2)
    else:
        p1_cmd_alt = np.full_like(ts, np.nan)
        p2_cmd_alt = np.full_like(ts, np.nan)
    
    # Use /exp topics if available, fallback to /mpa_cmd
    if len(t_p1_cmd) == 0:
        p1_cmd = p1_cmd_alt
    if len(t_p2_cmd) == 0:
        p2_cmd = p2_cmd_alt
    
    # Measured pressures from sensor
    if len(t_press_sens) > 0:
        p1_meas = nearest_interp(t_ref, t_press_sens, v_press_p1)
        p2_meas = nearest_interp(t_ref, t_press_sens, v_press_p2)
        ps_meas = p1_meas + p2_meas
        pd_meas = p1_meas - p2_meas
    else:
        print("[WARN] Pressure sensor data (/mpa_pressure) not found. Will fill with NaN.")
        p1_meas = np.full_like(ts, np.nan)
        p2_meas = np.full_like(ts, np.nan)
        ps_meas = np.full_like(ts, np.nan)
        pd_meas = np.full_like(ts, np.nan)
    
    # Joint angle
    if len(t_joint) > 0:
        theta = nearest_interp(t_ref, t_joint, v_joint)
        theta_deg = np.degrees(theta)
    else:
        print("[WARN] Joint data not found. Will fill with NaN.")
        theta = np.full_like(ts, np.nan)
        theta_deg = np.full_like(ts, np.nan)
    
    # Height: absolute z and relative dz
    if len(t_height) > 0:
        z_aligned = nearest_interp(t_ref, t_height, z_height)
        z0 = float(z_height[0])  # Initial height
        dz = z_aligned - z0
        print(f"[INFO] Initial height z0 = {z0:.6f} m")
    else:
        print("[WARN] Height data not found. Will fill z and dz with NaN.")
        z_aligned = np.full_like(ts, np.nan)
        dz = np.full_like(ts, np.nan)
    
    # Compute derivatives (for NARX features)
    dt = np.median(np.diff(ts)) if len(ts) > 1 else 0.005
    
    def gradient_safe(x, dt):
        """Compute gradient with forward/backward diff at boundaries"""
        if len(x) < 2:
            return np.zeros_like(x)
        dx = np.gradient(x, dt)
        return dx
    
    dps_dt = gradient_safe(ps, dt)
    dpd_dt = gradient_safe(pd, dt)
    
    # Build DataFrame
    df = pd.DataFrame({
        "t[s]": ts,
        "p1_cmd[MPa]": p1_cmd,
        "p2_cmd[MPa]": p2_cmd,
        "p1_meas[MPa]": p1_meas,
        "p2_meas[MPa]": p2_meas,
        "p_sum_cmd[MPa]": ps,
        "p_diff_cmd[MPa]": pd,
        "p_sum_meas[MPa]": ps_meas,
        "p_diff_meas[MPa]": pd_meas,
        "dp_sum_dt[MPa/s]": dps_dt,
        "dp_diff_dt[MPa/s]": dpd_dt,
        "theta[rad]": theta,
        "theta[deg]": theta_deg,
        "z[m]": z_aligned,
        "dz[m]": dz
    })
    
    # Save CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.8f")
    print(f"[OK] Wrote {len(df)} rows to {args.csv}")
    
    # Summary statistics
    print("\n=== Data Summary ===")
    print(f"Time span: {ts[0]:.3f} - {ts[-1]:.3f} s (duration: {ts[-1]-ts[0]:.3f} s)")
    print(f"Sampling rate: {1.0/dt:.1f} Hz (median dt = {dt*1000:.2f} ms)")
    print(f"\nPressure ranges (commanded):")
    print(f"  p1: [{np.nanmin(p1_cmd):.4f}, {np.nanmax(p1_cmd):.4f}] MPa")
    print(f"  p2: [{np.nanmin(p2_cmd):.4f}, {np.nanmax(p2_cmd):.4f}] MPa")
    print(f"  ps: [{np.nanmin(ps):.4f}, {np.nanmax(ps):.4f}] MPa")
    print(f"  pd: [{np.nanmin(pd):.4f}, {np.nanmax(pd):.4f}] MPa")
    
    if not np.all(np.isnan(p1_meas)):
        print(f"\nPressure ranges (measured):")
        print(f"  p1: [{np.nanmin(p1_meas):.4f}, {np.nanmax(p1_meas):.4f}] MPa")
        print(f"  p2: [{np.nanmin(p2_meas):.4f}, {np.nanmax(p2_meas):.4f}] MPa")
        print(f"  ps: [{np.nanmin(ps_meas):.4f}, {np.nanmax(ps_meas):.4f}] MPa")
        print(f"  pd: [{np.nanmin(pd_meas):.4f}, {np.nanmax(pd_meas):.4f}] MPa")
    
    if not np.all(np.isnan(theta)):
        print(f"\nTheta range: [{np.nanmin(theta_deg):.2f}, {np.nanmax(theta_deg):.2f}] deg")
    
    if not np.all(np.isnan(dz)):
        print(f"dz range: [{np.nanmin(dz):.6f}, {np.nanmax(dz):.6f}] m")
    
    # Data quality checks
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("\n[WARN] NaN values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count}/{len(df)} ({100*count/len(df):.1f}%)")


if __name__ == "__main__":
    main()