import mujoco
# import mujoco.viewer # 如果需要交互式查看器，但这里不需要
import imageio
import numpy as np
import os
import time

# --- MuJoCo MP4 Rendering ---
def create_simple_mujoco_xml(filepath="simple_model.xml"):
    """创建一个简单的 MuJoCo XML 文件用于演示"""
    xml_content = """
    <mujoco model="falling_box">
      <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
      <option integrator="RK4" timestep="0.01"/>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"
              margin="0.01" rgba="0.8 0.6 0.4 1"/>
      </default>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 0" size="2 2 0.1" type="plane" rgba="0.9 0.9 0.9 1"/>
        <body name="box_body" pos="0 0 1">
          <joint name="free_joint" type="free"/>
          <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="0 0.5 1 1" mass="1"/>
          <camera name="fixed_cam" pos="0 -1.5 1.5" xyaxes="1 0 0 0 0.707 0.707"/>
        </body>
      </worldbody>
    </mujoco>
    """
    with open(filepath, 'w') as f:
        f.write(xml_content)
    return filepath

def render_mujoco_to_mp4(model_path, output_path="mujoco_render.mp4", duration=5, fps=30, width=640, height=480, camera_name=None):
    """
    使用 MuJoCo 渲染并保存为 MP4。
    MuJoCo 3.3.2 使用新的 Renderer API。
    """
    print(f"MuJoCo: Loading model from {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        print("Make sure you have a valid MuJoCo XML file at the specified path.")
        return

    data = mujoco.MjData(model)

    print(f"MuJoCo: Initializing renderer ({width}x{height})")
    renderer = mujoco.Renderer(model, height, width)

    frames = []
    sim_time = 0.0
    frame_time_step = 1.0 / fps

    print(f"MuJoCo: Simulating for {duration} seconds, rendering at {fps} FPS...")
    start_render_time = time.time()

    # 确定使用哪个相机
    if camera_name:
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id == -1:
                print(f"Warning: Camera '{camera_name}' not found. Using default free camera.")
                camera_to_use = -1 # 默认自由相机
            else:
                camera_to_use = camera_id
                print(f"Using camera: {camera_name} (ID: {camera_id})")
        except Exception as e:
            print(f"Error getting camera ID for '{camera_name}': {e}. Using default free camera.")
            camera_to_use = -1
    else:
        print("No camera specified, using default free camera.")
        camera_to_use = -1 # 默认自由相机


    last_frame_sim_time = -frame_time_step # 确保第一帧被渲染
    while sim_time < duration:
        # 如果模拟时间超过了下一帧应渲染的时间，则渲染
        if sim_time - last_frame_sim_time >= frame_time_step:
            # 更新场景，指定相机（如果相机是固定的，可以只在循环外设置一次，但这里为了通用性每次都更新）
            # 对于自由相机，可以通过 renderer.scene.camera.lookat, .distance, .azimuth, .elevation 来控制
            # 如果使用XML中定义的相机，可以通过 camera_id 来指定
            renderer.update_scene(data, camera=camera_to_use) # camera=-1 使用自由相机
            pixels = renderer.render()
            frames.append(pixels)
            last_frame_sim_time = sim_time
            print(f"MuJoCo: Rendered frame at sim_time {sim_time:.2f}s, Num frames: {len(frames)}", end='\r')

        mujoco.mj_step(model, data)
        sim_time = data.time

    print(f"\nMuJoCo: Simulation complete. Total frames: {len(frames)}")
    end_render_time = time.time()
    print(f"MuJoCo: Rendering took {end_render_time - start_render_time:.2f} seconds.")

    if frames:
        print(f"MuJoCo: Saving video to {output_path}...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"MuJoCo: Video saved successfully to {output_path}")
    else:
        print("MuJoCo: No frames rendered, video not saved.")

    # 清理渲染器（可选，Python垃圾回收会处理，但显式关闭是个好习惯）
    renderer.close()


# --- Brax MP4 Rendering ---
# Brax 依赖 JAX
import jax
import jax.numpy as jnp
from brax import envs
from brax.io import image as brax_image # 用于渲染图像帧
class SimpleQP:
    def __init__(self, q, qd):
        self.q = q
        self.qd = qd
        # 下面这几个属性可选，设置为None或空也行
        self.qdd = None
        self.act = None
# from brax.io import html as brax_html # 另一种渲染方式，输出HTML
# from brax.io import model as brax_model # 如果加载自定义模型
def render_brax_to_mp4(env_name="ant", output_path="brax_render.mp4", duration=5, fps=30, width=640, height=480, seed=0):
    """
    使用新版 Brax 渲染并保存为 MP4。
    """
    print(f"Brax: Initializing environment '{env_name}'")
    try:
        env = envs.get_environment(env_name=env_name)
    except Exception as e:
        print(f"Error initializing Brax environment '{env_name}': {e}")
        print("Make sure the environment name is correct and Brax is installed properly.")
        return

    rng = jax.random.PRNGKey(seed)
    reset_key, step_key = jax.random.split(rng)

    state = env.reset(reset_key)
    jit_step = jax.jit(env.step)

    frames = []
    sim_time = 0.0
    num_steps = int(duration * fps)

    print(f"Brax: Simulating for {num_steps} steps (approx {duration}s at {fps} FPS)...")
    start_render_time = time.time()

    for i in range(num_steps):
        step_key, action_key = jax.random.split(step_key)
        action = jax.random.uniform(action_key, (env.action_size,), minval=-0.5, maxval=0.5)

        state = jit_step(state, action)

        # 使用 env.render 而不是 render_array + state.pipeline_state.qp
        rendered_frame = brax_image.render_array(
            sys=env.sys,
            trajectory=state.pipeline_state,
            width=width,
            height=height
        )
        frames.append(np.array(rendered_frame))  # 转为 NumPy 数组

        sim_time += env.dt
        print(f"Brax: Rendered frame {i+1}/{num_steps} at sim_time {sim_time:.2f}s", end='\r')

    print(f"\nBrax: Simulation complete. Total frames: {len(frames)}")
    end_render_time = time.time()
    print(f"Brax: Rendering took {end_render_time - start_render_time:.2f} seconds.")

    if frames:
        print(f"Brax: Saving video to {output_path}...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Brax: Video saved successfully to {output_path}")
    else:
        print("Brax: No frames rendered, video not saved.")


if __name__ == "__main__":
    # --- MuJoCo Demo ---
    print("--- Running MuJoCo Rendering Demo ---")
    mujoco_xml_file = create_simple_mujoco_xml("temp_mujoco_model.xml")
    render_mujoco_to_mp4(
        model_path=mujoco_xml_file,
        output_path="mujoco_falling_box.mp4",
        duration=3,  # 秒
        fps=30,
        width=320,
        height=240,
        camera_name="fixed_cam" # 使用XML中定义的相机
        # camera_name=None # 取消注释则使用默认自由相机
    )
    # 清理临时 XML 文件
    if os.path.exists(mujoco_xml_file):
        os.remove(mujoco_xml_file)
    print("-" * 30)
    print("\n")

    # --- Brax Demo ---
    print("--- Running Brax Rendering Demo ---")
    render_brax_to_mp4(
        env_name="ant", # 尝试 "humanoid", "fetch", "grasp", "halfcheetah" 等
        output_path="brax_ant_random_walk.mp4",
        duration=3,  # 秒
        fps=30,
        width=320,
        height=240,
        seed=42
    )
    print("-" * 30)

    print("\nAll rendering demos finished.")