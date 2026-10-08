# MDK practical assignment

The goal of this assignment is to make the end-effector of a robot arm track the full 6D pose of a moving target. You will do so for a Universal Robots UR5e arm simulated in the MuJoCo physics engine, and along the way you will build a small Python library for the geometric kinematics and control of serial-chain robots.

The full assignment — the tasks and the submission — is described in `assignment.pdf`. This README repeats its introduction and serves as a quick reference for setting up the project and running the scripts.

## Introduction

The differential kinematics of a serial robot is described by its geometric Jacobian. Given a coordinate frame $\Psi_c$, the geometric Jacobian in that frame is the linear map

```math
J^c(q)\colon T_q\mathcal{Q} \to \mathfrak{se}(3), \qquad
\dot q \mapsto J^c(q)\,\dot q = T_{\mathrm{ee}}^{c,\mathrm{w}},
```

that takes the joint velocities $\dot q$ to the twist $T_{\mathrm{ee}}^{c,\mathrm{w}}$ of the end-effector relative to the world, expressed in $\Psi_c$.

A common way to control the motion of the end-effector, for instance to track a trajectory, is to compute the desired twist of the end-effector and map it to joint velocities with the inverse of the Jacobian. This requires the Jacobian to be invertible, which is not always the case. If the robot does not have exactly six degrees of freedom, the Jacobian is not even square. Moreover, at certain configurations, called singularities, the Jacobian loses rank. Think of an arm stretched out straight that is asked to move its end-effector further in the direction of the stretch: no joint motion can achieve this, yet near such a configuration the inverse of the Jacobian returns very large joint velocities, which makes the controller unstable and the robot unsafe.

Instead of inverting the Jacobian, one can use its linear-algebraic dual,

```math
\bigl(J^c(q)\bigr)^*\colon \mathfrak{se}^*(3) \to T^*_q\mathcal{Q}, \qquad
W^c \mapsto W^c J^c(q) = \tau,
```

which gives the joint torques $\tau$ whose effect on the robot is equivalent to a wrench $W^c$ acting on the end-effector. Unlike the inverse, the dual map is defined in every configuration, singular or not, as well as for non-square Jacobians.

This suggests the following control law. From the pose $H_{\mathrm{ee}}^{\mathrm{d}}$ of the end-effector relative to the desired frame $\Psi_{\mathrm{d}}$, compute the wrench $W^{\mathrm{ee}}$ that a virtual spring connecting the two frames would exert on the end-effector, and command the joint torques obtained from this wrench through the dual of the Jacobian $J^{\mathrm{ee}}$ expressed in the end-effector frame. The robot then behaves as if its end-effector were pulled towards the target by the spring. Since a spring alone would make the end-effector oscillate around the target, a virtual damper is added, and the gravity and the other bias forces of the arm are compensated. The inverse kinematics problem is never solved: the controller works with the dual problem, mapping forces rather than velocities. As a result, it remains well-defined at singularities, and a target out of reach is harmless, as the arm simply stretches towards it.

You will develop a simple Python library for the geometric wrench control of a serial-chain robot step by step: you will describe the kinematics of the robot, compute its forward kinematics and geometric Jacobian, compute the virtual wrench, and map it to the joint torques. You will then test the controller on the UR5e arm in MuJoCo, first on a static target and then on a moving one.

> ⚠️ **Warning on the use of AI tools.** We are well aware that today's AI models, capable of solving open mathematical problems, can effortlessly complete this assignment. We nevertheless urge you to make your own effort to understand and solve the assignment yourself. While we are strictly against using AI tools to generate the solutions instead of you, we welcome their use for discussion: asking "why" questions, asking to explain the choices made in the code we provide, asking for guidance on Python syntax, etc. In your prompts, you may explicitly instruct the model not to give you a solution, but to help you understand the problem. What you learn by completing this assignment will greatly help you in the final exam.

## Installation

Everything is installed inside a self-contained conda environment: an isolated space with a specific version of Python and all the required packages, which does not interfere with what is already installed on your machine.

### 1. Install an IDE (optional)

You will be editing Python files and running them from a terminal. An IDE (integrated development environment) makes this considerably easier. Both of these widely used IDEs are free, and either is a good choice:

- **Visual Studio Code** — <https://code.visualstudio.com>. After installing it, add the **Python** extension by Microsoft (Extensions panel on the left).
- **PyCharm** — <https://www.jetbrains.com/pycharm/>.

The commands below are typed into a terminal; we recommend the one built into the IDE. A plain system terminal works just as well.

### 2. Get the project and open it in the IDE

Go to <https://github.com/vasilysent/robotics_course>, click the green **Code** button and then **Download ZIP**. Unzip the downloaded file and copy the `assignment` folder to a convenient location.

Open the `assignment` **folder** — the one containing `README.md` and `environment.yml` — with *File → Open Folder* in VS Code or *File → Open* in PyCharm. Then open a terminal inside the IDE: *Terminal → New Terminal* in VS Code, or the *Terminal* tab at the bottom of the PyCharm window. That terminal starts in the project folder. Check that it is the right one — the listing must contain `environment.yml`:

```bash
ls        # macOS / Linux
dir       # Windows
```

If it does not, go to the assignment folder with `cd path/to/the/assignment`.

### 3. Install the conda package manager

`conda` is the package manager that builds the environment. Check whether you already have it:

```bash
conda --version
```

If this prints a version number, such as `conda 26.3.2`, continue with step 4. If the command is not found, install **Miniforge**, a minimal conda installer set up with the **conda-forge** package channel, which is the channel this project's `environment.yml` asks for. Download it and follow the instructions for your operating system in its README:

<https://github.com/conda-forge/miniforge>

> **On an Intel Mac**, use Miniforge and not Miniconda: Anaconda stopped building for Intel Macs (`osx-64`) in August 2025, while Miniforge still ships an Intel installer.

When the installation has finished, **close the terminal and open a new one**, and check `conda --version` again. If it is still not found, the terminal does not know about conda yet:

- **macOS / Linux**: run `conda init zsh` (macOS) or `conda init bash` (Linux) from the folder Miniforge was installed in, then close the terminal and open a new one.
- **Windows**: the installer deliberately leaves `conda` off the search path, so only one shell knows about it: **Miniforge Prompt**, from the Start menu. (With Anaconda or Miniconda, the equivalent is called *Anaconda Prompt*.) To teach the terminal of the IDE about `conda` as well:

  1. Open **Miniforge Prompt** and run

     ```bash
     conda init powershell
     ```

     This writes a PowerShell startup file that makes `conda` available. Add `conda init cmd.exe` if you prefer the Command Prompt.

  2. Open a new terminal in the IDE and run

     ```powershell
     Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

     answering `Y` if prompted.

  3. **Restart the IDE.** The terminal now recognises `conda`.

### 4. Create the environment

In the terminal, in the project folder:

```bash
conda env create -f environment.yml
```

This creates an environment named `mdk-assignment` with Python 3.11 and installs the project into it together with everything it requires: MuJoCo, NumPy, PyYAML, imageio and matplotlib. It downloads a few hundred megabytes and takes a few minutes. It has to be done only once.

If it fails with `EnvironmentFileNotFound`, you are not in the project folder; go there with `cd path/to/the/project`.

### 5. Activate the environment

```bash
conda activate mdk-assignment
```

The command prompt now begins with `(mdk-assignment)`, which is how you can recognise that the environment is active.

### 6. Check that it worked

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

This should print `3.10.0`. Then run the first script:

```bash
python scripts/01_view_robot.py       # Linux / Windows
mjpython scripts/01_view_robot.py     # macOS
```

An interactive viewer should open, showing the UR5e arm in its reference (`q = 0`) configuration with its end-effector frame drawn at the flange. In the **Control** panel on the right, move the sliders to change the joint angles.

**On macOS, the scripts that open the viewer must be run with `mjpython` rather than `python`.** The viewer needs the main thread of the process to create its window; `mjpython` is a launcher that ships with MuJoCo and provides it. The scripts that only record a video open no window and are run with `python` on every platform.

### 7. Look at the project structure

The project follows the standard *src layout* of a Python project: the code lives in `src/assignment`, which makes `assignment` an installable library package, described by `pyproject.toml` and `requirements.txt`. The environment you created installs this package, so the scripts can `import assignment` from anywhere. The library code, the scripts that run it, and the simulation models are kept in separate folders:

```
models/                  MuJoCo models of the UR5e
scripts/                 the numbered scripts you run
src/assignment/          the library code provided
src/assignment/student/  the files you complete
```

Everything you need to write is in the `student` folder: the robot description `ur5e_kinematics.yaml`, the Lie group functions in `lie_groups.py`, the kinematics in `kinematics.py`, and the controller in `control.py`. The settings of the controller, such as the target and the gains, are in `config.py` in the same folder. The rest of the project is provided and does not need to be changed. Since the project is installed in editable mode, your changes take effect the next time you run a script.

## When you come back

If the command prompt does not begin with `(mdk-assignment)`, reactivate the environment from the project folder:

```bash
conda activate mdk-assignment
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| `conda: command not found` / `'conda' is not recognized` | See step 3. |
| `ModuleNotFoundError: No module named 'mujoco'` or `'assignment'` | The environment is not active; see step 5. |
| On macOS: `On macOS, run this script with:  mjpython ...` | Do exactly that, as in step 6. |
| On Windows: the terminal of the IDE does not recognise `conda`, although **Miniforge Prompt** does | Follow the three points for Windows in step 3. If the IDE opens PowerShell 7 (`pwsh`) rather than Windows PowerShell, run `conda init powershell` from inside `pwsh` too: the two keep separate startup files. |
| The IDE underlines the imports in red, or its Run button fails, while the same script works in the terminal | The IDE uses a different Python interpreter. VS Code: `Ctrl+Shift+P`, *Python: Select Interpreter*, choose `mdk-assignment`. PyCharm: *Settings → Project → Python Interpreter → Add → Conda Environment*, select the existing `mdk-assignment`. |
| `EnvironmentFileNotFound: 'environment.yml' file not found` | You are in the wrong folder; see step 4. |
| Importing `assignment` fails after you moved or renamed the project folder | Run `pip install -e .` from the project folder, with the environment active. |

## Updating and starting over

If the dependencies change, bring the environment up to date with:

```bash
conda env update -f environment.yml --prune
```

To delete the environment and build it from scratch:

```bash
conda deactivate
conda env remove -n mdk-assignment
conda env create -f environment.yml
```

## Assignment steps

An overview; the details are in `assignment.pdf`.

| Step | You complete | Check with |
|---|---|---|
| Robot description | `ur5e_kinematics.yaml`: the reference poses `H_ref` and unit twists of the joints, `H_0_w` and `H_ee_n` | `02_check_frames_reference.py` |
| Forward kinematics and Jacobian | `update_kinematics` in `kinematics.py`, with the Lie group functions in `lie_groups.py` | `03_check_forward_kinematics.py` |
| Virtual wrench | `spring_wrench`, `clip_wrench` and `control_wrench` in `control.py` | — |
| Control torques | `control_torque` in `control.py` | `04`, `05` and `06` |

## Running the scripts

With the environment active, from the project folder.

```bash
python scripts/01_view_robot.py                   # Linux / Windows
mjpython scripts/01_view_robot.py                 # macOS
```

The UR5e in the viewer with its end-effector frame. Use the sliders in the **Control** panel to find the positive direction of each joint.

```bash
python scripts/02_check_frames_reference.py       # Linux / Windows
mjpython scripts/02_check_frames_reference.py     # macOS
```

The link frames of `ur5e_kinematics.yaml`, drawn at the poses they have in the reference configuration. They stay fixed when the arm is moved with the sliders.

```bash
python scripts/03_check_forward_kinematics.py     # Linux / Windows
mjpython scripts/03_check_forward_kinematics.py   # macOS
```

The frames in the current configuration, as computed by your `update_kinematics`. If the description and the kinematics are correct, the frames stay attached to the links for every configuration.

```bash
python scripts/04_track_static_target.py          # Linux / Windows
mjpython scripts/04_track_static_target.py        # macOS
```

The arm, driven by joint torques, moves its end-effector to a static target frame and stays there. The simulation may run slower than real time on a slow machine. The target, the stiffnesses, the damping coefficients and the limits of the wrench are set in `src/assignment/student/config.py`. Try moving the target out of reach: the arm stretches towards it and remains stable.

```bash
python scripts/05_record_static_target.py         # every platform
```

The same motion, recorded to a video in true real time. Each run is saved in a timestamped subfolder of `runs/`, together with a plot of the tracking error and a copy of the `config.py` that produced it:

```
runs/2026-09-07_143012_static_target/
    video.mp4
    error.png
    config.py
```

```bash
python scripts/06_record_figure_eight.py          # every platform
```

The controller tracking a moving target that follows a figure-eight trajectory with a fixed orientation, recorded to a video with an error plot in the same way. The trajectory settings are in `config.py`.
