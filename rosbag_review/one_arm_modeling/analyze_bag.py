#!/usr/bin/env python3
#python analyze_bag.py   --bag 2025-09-12-19-27-02.bag   --sum-topic /exp_diff_grid_mpa/p_sum_MPa   --diff-topic /exp_diff_grid_mpa/p_diff_MPa   --joint-topic /kinikun1/joint_states   --joint-name arm1_joint   --cmd-topic /mpa_cmd   --cmd-scale 0.9/4096 --out-prefix out/diff_run1#
#csv, 基本的なプロットの作成
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
    # 1) ヒントそのまま
    if joint_topic_hint in topics: return joint_topic_hint
    # 2) 末尾/部分一致
    jt=find_topic(joint_topic_hint, topics)
    if jt: return jt
    # 3) JointState 型候補から name を覗いて選ぶ
    js_cands=[t for t,meta in topics.items() if meta["type"]=="sensor_msgs/JointState" and meta["count"]>0]
    for cand in js_cands:
        cnt=0
        with rosbag.Bag(bag._filename,"r") as b2:
            for topic,msg,stamp in b2.read_messages(topics=[cand]):
                # 属性があるかで判定（isinstanceを使わない）
                if hasattr(msg,"name") and hasattr(msg,"position"):
                    if joint_name in getattr(msg,"name",[]):
                        return cand
                cnt+=1
                if cnt>=10: break
    return js_cands[0] if js_cands else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--sum-topic",  default="/exp_diff_grid_mpa/p_sum_MPa")
    ap.add_argument("--diff-topic", default="/exp_diff_grid_mpa/p_diff_MPa")
    ap.add_argument("--joint-topic", default="/kinikun1/joint_states")
    ap.add_argument("--joint-name",  default="arm1_joint")
    ap.add_argument("--cmd-topic",   default="/mpa_cmd")
    ap.add_argument("--cmd-scale",   default=str(0.9/4096.0))
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

    sum_topic = find_topic(args.sum_topic, topics)
    diff_topic= find_topic(args.diff_topic, topics)
    joint_topic = pick_joint_topic(bag, topics, args.joint_topic, args.joint_name)
    cmd_topic = find_topic(args.cmd_topic, topics)

    print("=== resolved ===")
    print("sum_topic :", sum_topic)
    print("diff_topic:", diff_topic)
    print("joint_topic:", joint_topic)
    print("cmd_topic :", cmd_topic)

    if not sum_topic or not diff_topic:
        raise RuntimeError("sum/diff が見つかりません。--list で確認してください。")

    # 収集
    t_sum,v_sum=[],[]
    t_diff,v_diff=[],[]
    t_joint,v_joint=[],[]
    t_cmd,p1_cmd,p2_cmd=[],[],[]

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

    bag.close()

    # チェック
    if len(t_sum)==0 or len(t_diff)==0:
        raise RuntimeError(f"sum/diff が空です。sum:{len(t_sum)} diff:{len(t_diff)}")
    if len(t_joint)==0:
        raise RuntimeError(f"JointState 読み取り0件。joint_topic={joint_topic} joint_name={args.joint_name}。--list で件数を確認してください。")

    # 整列（和圧基準）
    t_sum=np.asarray(t_sum); v_sum=np.asarray(v_sum)
    t_diff=np.asarray(t_diff); v_diff=np.asarray(v_diff)
    t_joint=np.asarray(t_joint); v_joint=np.asarray(v_joint)
    t_cmd=np.asarray(t_cmd); p1_cmd=np.asarray(p1_cmd); p2_cmd=np.asarray(p2_cmd)

    t0=min(t_sum[0], t_joint[0])
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

    # CSV
    csv_path=args.out_prefix+"_data.csv"
    header="t[s],p_sum[MPa],p_diff[MPa],p1[MPa],p2[MPa],p1_cmd[MPa],p2_cmd[MPa],theta[rad],theta[deg]"
    np.savetxt(csv_path, np.column_stack([ts,ps,pd,p1,p2,p1c,p2c,th,th_deg]),
               delimiter=",", header=header, comments="")
    print("[OK] wrote", csv_path)

    # 図
    plt.figure(figsize=(11,6))
    plt.plot(ts, ps, label="p_sum [MPa]")
    plt.plot(ts, pd, label="p_diff [MPa]")
    plt.plot(ts, p1, label="p1 [MPa] (sum/diff)")
    plt.plot(ts, p2, label="p2 [MPa] (sum/diff)")
    if not np.all(np.isnan(p1c)):
        plt.plot(ts, p1c, "--", label="p1_cmd [MPa]")
        plt.plot(ts, p2c, "--", label="p2_cmd [MPa]")
    plt.plot(ts, th_deg, label="theta [deg]")
    plt.xlabel("time [s]"); plt.ylabel("value"); plt.grid(True); plt.legend(); plt.tight_layout()
    f1=args.out_prefix+"_timeseries.png"; plt.savefig(f1, dpi=150); print("[OK] wrote", f1)

    plt.figure(figsize=(7,6))
    sc=plt.scatter(pd, th_deg, c=ps, s=6); plt.colorbar(sc, label="p_sum [MPa]")
    plt.xlabel("p_diff [MPa]"); plt.ylabel("theta [deg]"); plt.grid(True); plt.tight_layout()
    f2=args.out_prefix+"_scatter_theta_vs_pdiff.png"; plt.savefig(f2, dpi=150); print("[OK] wrote", f2)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.scatter(p1, th_deg, s=6); plt.xlabel("p1 [MPa]"); plt.ylabel("theta [deg]"); plt.title("theta vs p1"); plt.grid(True)
    plt.subplot(1,2,2); plt.scatter(p2, th_deg, s=6); plt.xlabel("p2 [MPa]"); plt.ylabel("theta [deg]"); plt.title("theta vs p2"); plt.grid(True)
    plt.tight_layout()
    f3=args.out_prefix+"_scatter_theta_vs_p1p2.png"; plt.savefig(f3, dpi=150); print("[OK] wrote", f3)

    print("[DONE]")

if __name__=="__main__":
    main()
