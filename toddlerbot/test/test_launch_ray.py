import mujoco
import mujoco.viewer
import time
import os
import numpy as np

# cast_ray 函数保持不变
def cast_ray(model, data, pnt_world: np.ndarray, vec_world: np.ndarray, bodyexclude: int = -1):
    pnt = np.reshape(pnt_world.astype(np.float64), (3, 1))
    vec = np.reshape(vec_world.astype(np.float64), (3, 1))
    geomid_arr = np.zeros((1, 1), dtype=np.int32)
    dist_factor = mujoco.mj_ray(
        model, data, pnt, vec,
        geomgroup=None, flg_static=1,
        bodyexclude=bodyexclude, geomid=geomid_arr
    )
    hit = (geomid_arr[0, 0] != -1)
    geom_id_hit = geomid_arr[0, 0] if hit else -1
    return hit, geom_id_hit, dist_factor


xml_path = 'temp_vertical_side_scan_model.xml' # 新文件名
if not os.path.exists(xml_path):
    xml_content =  """
    <mujoco model="vertical_side_scan_example">
        <option timestep="0.01" gravity="0 0 -0.81"/>
        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <map znear="0.001"/>
        </visual>
        <worldbody>
            <light diffuse=".7 .7 .7" pos="0 0 4" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" pos="0 0 -0.5"/>
            
            <!-- 物体放在球体扫描路径上 (左右和下方) -->
            <body name="target_left_low" pos="0 -0.5 0"> <!-- 球体左方偏下 -->
                <geom name="left_sphere" type="sphere" size="0.15" rgba="0 1 0 1"/>
            </body>
            <body name="target_directly_below" pos="0 0 -0.2"> <!-- 球体正下方 -->
                <geom name="blow_box" type="box" size="0.2 0.05 0.2" rgba="0 0.5 1 1"/> <!-- xz 大小调换 -->
            </body>
            <body name="target_right_low" pos="0 0.5 0"> <!-- 球体右方偏下 -->
                <geom name="right_capsule" type="capsule" fromto="0 0 -0.1 0 0 0.1" size="0.1" rgba="1 0.5 0 1"/>
            </body>

            <body name="ball" pos="0 0 1.0">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.1" rgba="1 0 0 1" mass="0.1"/>
                <site name="ray_origin_vis" type="sphere" size="0.02" rgba="1 1 0 0.8" pos="0 0 -0.1"/>
                <!-- Sites for visualization -->
                <site name="ball_local_y_marker" type="box" size="0.01 0.1 0.01" pos="0 0.11 0" rgba="0 1 1 0.5"/> <!-- 标记球的局部+Y方向 -->
                <site name="ray_hit_vis_0" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_1" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_2" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_3" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_4" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_5" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_6" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_7" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_8" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_9" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_10" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_11" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_12" type="sphere" size="0.01" rgba="1 0 1 0.7"/>
                <site name="ray_hit_vis_13" type="sphere" size="0.01" rgba="1 0 1 0.7"/>

            </body>
        </worldbody>
        <!-- 可选: 添加用于可视化射线击中点的site -->
        <!--
        ...
        -->
    </mujoco>
    """
    with open(xml_path, "w") as f:
        f.write(xml_content)

model = mujoco.MjModel.from_xml_path(xml_path)
if model is None: exit()
data = mujoco.MjData(model)

# --- 球体信息 ---
ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
if ball_geom_id == -1 and model.body_geomnum[ball_body_id] > 0:
    ball_geom_id = model.body_geomadr[ball_body_id]
ball_radius = model.geom_size[ball_geom_id, 0]

# 初始速度 (让球体运动和旋转)
if model.body_jntnum[ball_body_id] > 0:
    ball_jnt_id = model.body_jntadr[ball_body_id]
    if model.jnt_type[ball_jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
        ball_qvel_adr = model.jnt_dofadr[ball_jnt_id]
        data.qvel[ball_qvel_adr + 0] = 0.1  # X轴线速度
        # data.qvel[ball_qvel_adr + 1] = 0.05  # Y轴线速度
        # data.qvel[ball_qvel_adr + 2] = 0.5  # Z轴线速度
        data.qvel[ball_qvel_adr + 3] = 0.3 # 绕X轴旋转 (改变Y和Z的朝向)
        data.qvel[ball_qvel_adr + 5] = 0.2  # 绕Z轴旋转 (在XY平面旋转)


# --- 射线扫描参数 ---
num_rays = 11
# 扫描角度定义：0度=球体局部-Y轴（向左水平），90度=球体局部-Z轴（向下），180度=球体局部+Y轴（向右水平）
scan_angle_start_deg = 0.0
scan_angle_end_deg = 180.0

offset_from_surface = 0.02
ray_length_on_miss = 1.0

# --- 可视化 ---
MAX_VIS_RAYS = num_rays
ray_hit_vis_site_ids = []
# ... (与之前相同的 site 加载逻辑，如果需要，请在XML中添加 ray_hit_vis_i)
# 比如在XML的ball body内部添加：
# <site name="ray_hit_vis_0" type="sphere" size="0.01" rgba="0.8 0 0.8 0.7"/>
# ...
# <site name="ray_hit_vis_10" type="sphere" size="0.01" rgba="0.8 0 0.8 0.7"/>
for i in range(min(num_rays, MAX_VIS_RAYS)):
    site_name = f"ray_hit_vis_{i}"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id != -1:
        ray_hit_vis_site_ids.append(site_id)

ray_origin_vis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ray_origin_vis")


try:
    # 获取球体的 body ID (geom 的父 body)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if geom_id != -1:
        tracked_body_id = model.geom_bodyid[geom_id]
    else:
        print("Warning: Geom 'ball_geom' not found. Using world body for tracking.")
        tracked_body_id = 0 # 默认跟踪世界坐标系原点或者第一个物体

except Exception as e:
    print(f"Error finding body: {e}")
    tracked_body_id = 0 # Fallback

with mujoco.viewer.launch_passive(model, data) as viewer:
    simulation_duration_sim_time = 100.0
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY  # ✅ 显示 body 坐标系

    viewer.cam.lookat = [data.qpos[0], data.qpos[1], data.qpos[2]]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -10 # 稍微调整视角
    viewer.cam.azimuth = 0 # 正对球的Y轴方向
    if tracked_body_id > 0: # 确保不是 world body (ID 0)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING # 设置相机模式为跟踪模式
        viewer.cam.trackbodyid = tracked_body_id            # 设置要跟踪的 body ID
        # 或者，如果想让相机固定在一个物体上并看向另一个物体，可以使用：
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer.cam.fixedcamid = some_camera_id_defined_in_xml_targeting_the_body
    else:
        print("No valid body to track, using default camera settings.")
        
    loop_start_wall_time = time.time()
    print_interval_wall_time = 0.25
    last_print_wall_time = loop_start_wall_time
    frame_count_interval = 0

    scan_angles_rad = np.linspace(
        np.deg2rad(scan_angle_start_deg),
        np.deg2rad(scan_angle_end_deg),
        num_rays
    )

    print(f"将扫描 {num_rays} 条射线，在球体局部YZ平面，角度从 {scan_angle_start_deg:.1f} deg (局部-Y) 到 {scan_angle_end_deg:.1f} deg (局部+Y).")

    while viewer.is_running() and data.time < simulation_duration_sim_time:
        ball_center_world = data.xpos[ball_body_id].copy()
        ball_rotation_matrix_world = data.xmat[ball_body_id].reshape(3, 3).copy()

        local_origin_offset = np.array([0, 0, -(ball_radius + offset_from_surface)])
        pnt_world_origin = ball_center_world + ball_rotation_matrix_world @ local_origin_offset
        

        hit_details = []

        mujoco.mj_step(model, data)
        frame_count_interval += 1
        
        for i, angle_rad in enumerate(scan_angles_rad):
            # 射线方向在球体的局部YZ平面内定义:
            # angle_rad = 0 (0 deg): 指向局部-Y (左) -> [0, -1, 0] in local frame
            # angle_rad = pi/2 (90 deg): 指向局部-Z (下) -> [0, 0, -1] in local frame
            # angle_rad = pi (180 deg): 指向局部+Y (右) -> [0, 1, 0] in local frame
            
            # y_comp_local: angle=0 -> -1 (cos(pi)=-1), angle=pi/2 -> 0 (cos(pi/2)=0), angle=pi -> 1 (cos(0)=1)
            # z_comp_local: angle=0 -> 0 (sin(pi)=0), angle=pi/2 -> -1 (sin(pi/2)=1), angle=pi -> 0 (sin(0)=0)
            # 需要将 angle_rad 映射到 [pi, 0] 或调整三角函数
            # 或者，更直接地：
            # 水平分量 (沿Y轴) 使用 cos，垂直分量 (沿Z轴) 使用 sin
            # angle_rad 0度是-Y, 90度是-Z, 180度是+Y
            
            # 假设扫描角 'phi' 从 -pi/2 (局部-Y) 到 +pi/2 (局部+Y), 0 是局部-Z
            # phi = angle_rad - np.pi/2
            # y_comp_local = np.sin(phi)
            # z_comp_local = -np.cos(phi)

            # 更简单的映射：angle_rad 从 0 (局部-Y) 到 pi (局部+Y)
            # 中间点 pi/2 是 局部-Z
            # y_comp_local: angle_rad=0 -> -1, angle_rad=pi/2 -> 0, angle_rad=pi -> 1
            # z_comp_local: angle_rad=0 -> 0, angle_rad=pi/2 -> -1, angle_rad=pi -> 0
            
            # 使用cos(angle_rad)从-1到1，但我们需要调整使其匹配-Y到+Y
            # 使用cos(angle_rad) for y-component after shifting angle_rad
            # angle_rad_shifted for y: when angle_rad=0, effective_angle=pi, cos(pi)=-1.
            #                            when angle_rad=pi, effective_angle=0, cos(0)=1.
            # So, y_effective_angle = np.pi - angle_rad
            y_comp_local = -np.cos(angle_rad) # angle_rad=0 -> -1 (-Y); angle_rad=pi/2 -> 0; angle_rad=pi -> 1 (+Y)

            # z_comp_local: angle_rad=0 -> 0, angle_rad=pi/2 -> -1, angle_rad=pi -> 0
            z_comp_local = -np.sin(angle_rad) # angle_rad=0 -> 0; angle_rad=pi/2 -> -1 (-Z); angle_rad=pi -> 0

            vec_local_scan_plane = np.array([0, # 没有X分量
                                             y_comp_local,
                                             z_comp_local])
            
            vec_world = ball_rotation_matrix_world @ vec_local_scan_plane
            vec_world_normalized = vec_world / (np.linalg.norm(vec_world) + 1e-9)

            hit, geom_id_hit, dist_factor = cast_ray(model, data, pnt_world_origin, vec_world_normalized, bodyexclude=ball_body_id)
            actual_hit_distance = dist_factor

            hit_details.append({
                "angle_deg": np.rad2deg(angle_rad),
                "hit": hit,
                "geom_id": geom_id_hit,
                "distance": actual_hit_distance,
            })

            #打印激光雷达的投射点阵（紫色）
            if i < len(ray_hit_vis_site_ids):
                site_to_update_id = ray_hit_vis_site_ids[i]
                if hit:
                    intersection_point = pnt_world_origin + vec_world_normalized * actual_hit_distance
                    data.site_xpos[site_to_update_id] = intersection_point
                else:
                    data.site_xpos[site_to_update_id] = pnt_world_origin + vec_world_normalized * ray_length_on_miss
 
        #更新投射点原点
        if ray_origin_vis_id != -1:
            data.site_xpos[ray_origin_vis_id] = pnt_world_origin

        elapsed_sim_time_total = data.time
        elapsed_wall_time_total = time.time() - loop_start_wall_time
        time_to_sleep_for_rt = elapsed_sim_time_total - elapsed_wall_time_total
        if time_to_sleep_for_rt > 0:
            time.sleep(time_to_sleep_for_rt)
        
        current_wall_time = time.time()
        if current_wall_time - last_print_wall_time >= print_interval_wall_time:
            fps = frame_count_interval / (current_wall_time - last_print_wall_time)
            print(f"--- WallT: {current_wall_time - loop_start_wall_time:.2f}s, SimT: {data.time:.2f}s, FPS: {fps:.1f} ---")
            ball_y_axis_world = ball_rotation_matrix_world[:, 1] # 球的局部Y轴在世界中的方向
            print(f"Ball local Y-axis in world: [{ball_y_axis_world[0]:.2f}, {ball_y_axis_world[1]:.2f}, {ball_y_axis_world[2]:.2f}]")
            for detail in hit_details:
                if detail["hit"]:
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, detail["geom_id"])
                    print(f"  Angle(0=-Y,90=-Z,180=+Y): {detail['angle_deg']:.1f} deg, Hit: '{geom_name}', Dist: {detail['distance']:.3f}")
            last_print_wall_time = current_wall_time
            frame_count_interval = 0

        viewer.sync()

if xml_path == "temp_vertical_side_scan_model.xml": #确保删除正确的文件
    os.remove(xml_path)
print("仿真结束。")