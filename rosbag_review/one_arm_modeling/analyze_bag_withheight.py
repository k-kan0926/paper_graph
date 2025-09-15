#!/usr/bin/env python3
#python analyze_bag_withheight.py   --bag 2025-09-15-18-14-00.bag   --sum-topic /exp_diff_grid_mpa/p_sum_MPa   --diff-topic /exp_diff_grid_mpa/p_diff_MPa   --joint-topic /kinikun1/joint_states   --joint-name arm1_joint   --cmd-topic /mpa_cmd   --cmd-scale 0.9/4096   --height-topic /arm1/pose   --out-prefix out/diff_run1_h
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, matplotlib.pyplot as plt, rosbag

def to_secs(t): return t.to_sec() if hasattr(t,"to_sec") else float(t)

def nearest_interp(x_ref, x_src, y_src):
    if len(x_src)==0: return np.full_like(x_ref, np.nan, float)
    x_src=np.asarray(x_src); y_src=np.asarray(y_src); x_ref=np.asarray(x_ref)
    idx=np.searchsorted(x_src, x_ref, side="left"); idx=np.clip(idx,0,len(x_src)-1)
    mask=(idx>0) & ((idx==len(x_src)) | (np.abs(x_ref-x_src[idx-1])<=np.abs(x_ref-x_src[idx])))
    idx[mask]-=1; return y_src[idx]

def list_topics(bag):
    info=bag.get_type_and_topic_info().topics
    return {t:{"type":meta.msg_type,"count":meta.message_count} for t,meta in info.items()}

def find_topic(name_hint, topics):
    if name_hint in topics: return name_hint
    tails=[t for t in topics if t.endswith(name_hint)]
    if tails: return tails[0]
    parts=[t for t in topics if name_hint in t]
    return parts[0] if parts else None

def pick_joint_topic(bag, topics, joint_topic_hint, joint_name):
    if joint_topic_hint in topics: return joint_topic_hint
    jt=find_topic(joint_topic_hint, topics)
    if jt: return jt
    js_cands=[t for t,meta in topics.items() if meta["type"]=="sensor_msgs/JointState" and meta["count"]>0]
    for cand in js_cands:
        cnt=0
        with rosbag.Bag(bag._filename,"r") as b2:
            for topic,msg,stamp in b2.read_messages(topics=[cand]):
                if hasattr(msg,"name") and hasattr(msg,"position"):
                    if joint_name in getattr(msg,"name",[]):
                        return cand
                cnt+=1
                if cnt>=10: break
    return js_cands[0] if js_cands else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--sum-topic",   default="/exp_diff_grid_mpa/p_sum_MPa")
    ap.add_argument("--diff-topic",  default="/exp_diff_grid_mpa/p_diff_MPa")
    ap.add_argument("--joint-topic", default="/kinikun1/joint_states")
    ap.add_argument("--joint-name",  default="arm1_joint")
    ap.add_argument("--cmd-topic",   default="/mpa_cmd")     # geometry_msgs/Vector3 (x=p1_raw,y=p2_raw)
    ap.add_argument("--cmd-scale",   default=str(0.9/4096.0))
    # ★ 高さトピック（geometry_msgs/PoseStamped）
    ap.add_argument("--height-topic", default="/arm1/pose",
                    help="geometry_msgs/PoseStamped (pose.position.z を使用)")
    ap.add_argument("--out-prefix",  default="out/analysis")
    ap.add_argument("--list", action="store_true")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    try: scale=float(eval(args.cmd_scale,{},{}))
    except Exception: scale=float(args.cmd_scale)

    print(f"[INFO] reading bag: {args.bag}")
    bag=rosbag.Bag(args.bag,"r")
    topics=list_topics(bag)
    if args.list:
        print("=== topics ===")
        for t,meta in sorted(topics.items()):
            print(f"{t:60s} [{meta['type']}] x{meta['count']}")
        return

    sum_topic  = find_topic(args.sum_topic, topics)
    diff_topic = find_topic(args.diff_topic, topics)
    joint_topic= pick_joint_topic(bag, topics, args.joint_topic, args.joint_name)
    cmd_topic  = find_topic(args.cmd_topic, topics)
    height_topic = find_topic(args.height-topic if hasattr(args,"height-topic") else args.height_topic, topics)

    print("=== resolved ===")
    print("sum_topic   :", sum_topic)
    print("diff_topic  :", diff_topic)
    print("joint_topic :", joint_topic)
    print("cmd_topic   :", cmd_topic)
    print("height_topic:", height_topic)

    if not sum_topic or not diff_topic:
        raise RuntimeError("sum/diff が見つかりません。--list で確認してください。")
    if not joint_topic:
        raise RuntimeError("joint_states が見つかりません。--list で確認してください。")

    # 収集
    t_sum,v_sum=[],[]
    t_diff,v_diff=[],[]
    t_joint,v_joint=[],[]
    t_cmd,p1_cmd,p2_cmd=[],[],[]
    t_h, z_h = [], []   # height（絶対 z）

    for topic,msg,stamp in bag.read_messages():
        ts=to_secs(stamp)
        if topic==sum_topic and hasattr(msg,"data"):
            t_sum.append(ts); v_sum.append(float(msg.data))
        elif topic==diff_topic and hasattr(msg,"data"):
            t_diff.append(ts); v_diff.append(float(msg.data))
        elif topic==cmd_topic and hasattr(msg,"x") and hasattr(msg,"y"):
            t_cmd.append(ts); p1_cmd.append(float(msg.x)*scale); p2_cmd.append(float(msg.y)*scale)
        elif topic==joint_topic and hasattr(msg,"name") and hasattr(msg,"position"):
            names=list(msg.name)
            if args.joint_name in names:
                i=names.index(args.joint_name)
            else:
                i=0 if len(msg.position)>0 else None
            if i is not None:
                t_joint.append(ts); v_joint.append(float(msg.position[i]))
        elif height_topic and topic==height_topic and hasattr(msg,"pose") and hasattr(msg.pose,"position"):
            pos = msg.pose.position
            if hasattr(pos, "z"):
                t_h.append(ts); z_h.append(float(pos.z))

    bag.close()

    # numpy化
    t_sum=np.asarray(t_sum); v_sum=np.asarray(v_sum)
    t_diff=np.asarray(t_diff); v_diff=np.asarray(v_diff)
    t_joint=np.asarray(t_joint); v_joint=np.asarray(v_joint)
    t_cmd=np.asarray(t_cmd); p1_cmd=np.asarray(p1_cmd); p2_cmd=np.asarray(p2_cmd)
    t_h=np.asarray(t_h); z_h=np.asarray(z_h)

    if len(t_sum)==0 or len(t_diff)==0:
        raise RuntimeError(f"sum/diff が空です。sum:{len(t_sum)} diff:{len(t_diff)}")
    if len(t_joint)==0:
        raise RuntimeError(f"JointState 読み取り0件。joint_topic={joint_topic} joint_name={args.joint_name}")

    # 整列（和圧基準）
    t0=min(t_sum[0], t_joint[0])
    if len(t_h)>0: t0=min(t0, t_h[0])
    t_ref=t_sum; ts=t_ref - t0

    ps=v_sum
    pd=nearest_interp(t_ref, t_diff, v_diff)
    th=nearest_interp(t_ref, t_joint, v_joint)
    th_deg=np.degrees(th)

    p1=0.5*(ps+pd); p2=0.5*(ps-pd)

    if len(t_cmd)>0:
        p1c=nearest_interp(t_ref, t_cmd, p1_cmd); p2c=nearest_interp(t_ref, t_cmd, p2_cmd)
    else:
        p1c=np.full_like(ts, np.nan); p2c=np.full_like(ts, np.nan)

    # 高さ：絶対 z と Δz（最初の z を基準）
    if len(t_h)>0:
        z_aligned = nearest_interp(t_ref, t_h, z_h)
        z0 = float(z_h[0])  # 最初に観測した z
        dz = z_aligned - z0
    else:
        print("[warn] height_topic が見つからない/0件のため、z, dz は NaN で出力します。")
        z_aligned = np.full_like(ts, np.nan)
        dz = np.full_like(ts, np.nan)

    # CSV 追記（末尾に z[m], dz[m]）
    csv_path=args.out_prefix+"_data.csv"
    header="t[s],p_sum[MPa],p_diff[MPa],p1[MPa],p2[MPa],p1_cmd[MPa],p2_cmd[MPa],theta[rad],theta[deg],z[m],dz[m]"
    np.savetxt(csv_path, np.column_stack([ts,ps,pd,p1,p2,p1c,p2c,th,th_deg,z_aligned,dz]),
               delimiter=",", header=header, comments="")
    print("[OK] wrote", csv_path)

    # （任意）簡易プロットに高さを重ねる
    plt.figure(figsize=(11,6))
    plt.plot(ts, ps, label="p_sum [MPa]")
    plt.plot(ts, pd, label="p_diff [MPa]")
    plt.plot(ts, th_deg, label="theta [deg]")
    if np.isfinite(dz).any():
        plt.plot(ts, dz, label="dz [m]")
    plt.xlabel("time [s]"); plt.ylabel("value"); plt.grid(True); plt.legend(); plt.tight_layout()
    f1=args.out_prefix+"_timeseries_with_dz.png"; plt.savefig(f1, dpi=150); print("[OK] wrote", f1)

    print("[DONE]")

if __name__=="__main__":
    main()
