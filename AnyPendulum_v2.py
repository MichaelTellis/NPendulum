
'''
AnyPendulum_v2.py - smulates an N-link pendulum with arbitrary masses, lengths, and initial conditions from the command line.

Type settings as name=value to change the default settings. Seperate using spaces. They can be in any order, and will use default settings if not specified. 

Lists use commas with no spaces.

For example, to simulate a 2-link pendulum with starting angle of 90 degrees for the first link and 0 degrees for the second link, using the DOP853 solver, run:
    python AnyPendulum_v2.py links=2 thetas=90,0 methods=DOP853
Run  python AnyPendulum_v2.py help  to see every setting.
'''
import math
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib

from pathlib import Path
import sys
import warnings
from pprint import pprint


METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

# ------------------- Configuration -------------------
#default config
DEFAULTS = {
    #pendulum setup
    "links": (4, "Number of pendulum links"),
    "masses": ([1.0], "Mass of each link"),
    "lengths": ([1.0], "Length of each link"),
    "physical": (False, "Whether to use physical pendulum model"),
    "a_cm": ([0.4], "with physical=true: sideways offset of center of mass"),
    "b_cm": ([0.5], "with physical=true: downward offset of center of mass"),
    #initial conditions
    "thetas": ([0.0], "Initial angles of each link DEGREES"),
    "omegas": ([0.0], "Initial angular velocities of each link, RAD/SEC"),
    #physics and taylor approx
    "pivot": ("none", "Pivot type: none, other (a_x = -t), or earthquake"),
    "earthquake": ("real", "Earthquake type: real or fake"),
    "earthquake_data_folder": ("Earthquake Acceleration Data", "Earthquake data folder path"),
    "earthquake_specific": ("RSN31_PARKF_C08050.AT2", "Earthquake file path"),
    "damping": (0.3, "Damping coefficient (0 for no damping)"),
    "order": ("none", "none = exact sol. Specifies order of taylor approx"),
    "g": (9.81, "Gravitational acceleration"),
    #time
    "start": (0.0, "Start time (s)"),
    "end": (60.0, "End time (s)"),
    "framerate": (60, "calculations per simulated second, this is how many points per step are saved"),
    #integration
    "methods": (["DOP853"], "solvers to run: all or any in " + ",".join(METHODS)),
    "rtol": (1e-10, "Relative tolerance for solvers"),
    "atol": (1e-12, "Absolute tolerance for solvers"),
    #output
    "output": ("pendulum_output", "Output folder"),
    "save_images": (False, "Whether to save images and GIFs"),
    "gif": (False, "Whether to make GIFs, note: GIFs are only made if save_images is True as well"),
    "gif_framerate": (40, "Framerate for GIFs"),
    "trace_path": (True, "Whether to trace the path of the pendulum"),
    "run_phase_space": (True, "Whether to determine the phase space of the pendulum."),
    "playback": (0.5, "Playback speed for GIFs"),
    "follow_pivot": (True, "Whether to follow the pivot point in the animation"),
    "plot_show": (True, "Whether to show plots and animations")
}
TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no"}

# AI wrote the validation functions: stop and print_help(): which are used a lot later.
def print_help():
    print(__doc__)
    print("Settings (name=default):")
    for name, (default, description) in DEFAULTS.items():
        if isinstance(default, list):
            default = ",".join(str(v) for v in default)
        print(f"  {name + '=' + str(default):<32} {description}")

def stop(message):
    sys.exit(f"Error: {message}\nRun  python AnyPendulum_v2.py help  to see every setting.")

def get_settings(argv=None):
    settings = {name: default for name, (default, _) in DEFAULTS.items()}
    for arg in argv:
        if arg.lower() == "help":
            print_help()
            sys.exit(0)
        if "=" not in arg:
            stop(f"'{arg}' format is name=value, for example: links=3.")
        name, text = arg.split("=", 1)

        
        if name not in settings:
            stop(f"Unknown setting '{name}'.")
        settings[name] = convert(name, text, DEFAULTS[name][0])
    return settings

def convert(name, text, default):
    if isinstance(default,bool):
        if text.lower() in TRUE_VALUES:
            return True
        elif text.lower() in FALSE_VALUES:
            return False
        else:
            stop(f"Wrong boolean value for '{name}': {text}. Use true/false, 1/0, yes/no.")
    elif isinstance(default, list):     
        return [convert(name,piece_of_text, default[0]) for piece_of_text in text.split(",")]
    #AI wrote this following section of the if statement since otherwise I was checking every single setting and that is dumb.
    try:
        return type(default)(text)
    except (ValueError, TypeError):
        stop(f"Invalid value for '{name}': '{text}'. Expected type: {type(default).__name__}.")
    
# ------------------- Validation -------------------
def validate_settings(settings):
    N = settings["links"]
    if N < 1:
        stop("Number of links must be at least 1.")
    
    for arr_name in ["masses", "lengths", "thetas", "omegas", "a_cm", "b_cm"]:
        arr = settings[arr_name]
        if len(arr) == 1:
            arr = np.full(N, arr[0])

        elif len(arr) != N:
            stop(f"Length of '{arr_name}' must be equal to number of links ({N}).")
        settings[arr_name] = np.array(arr, dtype=float)
    settings["thetas"] = np.radians(settings["thetas"])  # Conv degree to rads
    if settings["order"] == "none":
        settings["order"] = None
    elif settings["order"].isdigit() and int(settings["order"]) >=1:
        settings["order"] = int(settings["order"])
    else:
        stop(f"Order is not valid, must be 'none' or a positive int.")

    if settings["pivot"] not in ("none", "other", "earthquake"):
        stop(f"Pivot type must be 'none', 'other', or 'earthquake'.")
    if settings["methods"] == ["all"]:
        settings["methods"] = METHODS
    else:
        for method in settings["methods"]:
            if method not in METHODS:
                stop(f"unknown integration method {method}. must be in this list: {METHODS}")
    if settings["start"] >= settings["end"]:
        stop("Start time must be less than end time.")
    return settings

def print_settings(settings):
    
    print("-" * 20 + "Pendulum Simulation Settings:" + "-" * 20 + "\n")
    pprint(settings, sort_dicts=False) 
    print("-" * 20 + "\n change settings by running: python AnyPendulum_v2.py name=value; run  python AnyPendulum_v2.py help to see list of settings \n")


settings = get_settings(sys.argv[1:])
settings = validate_settings(settings)

if settings["earthquake"] not in ("real", "fake"):
    stop("Earthquake must be 'real' or 'fake'.")

if settings["pivot"] == "earthquake" and settings["earthquake"] == "real":
    if not (Path(settings["earthquake_data_folder"]) / settings["earthquake_specific"]).is_file():
        stop("Earthquake file does not exist.")

print_settings(settings)
#--------- Get configuration -------------------
N = settings["links"]

masses = settings["masses"]
lengths = settings["lengths"]

initial_thetas = settings["thetas"]
initial_omegas = settings["omegas"]

ORDER =settings["order"]
PHYSICAL_PENDULUM = settings["physical"]

# Reset for if not physical but point masses
if PHYSICAL_PENDULUM:
    # Local coordinates: +a to the right, +b down when theta = 0
    I_cm = masses * lengths**2 / 12.0
    a_cm = settings["a_cm"]
    b_cm = settings["b_cm"]
else:
    I_cm = np.zeros(N)
    a_cm = np.zeros(N)
    b_cm = lengths


USE_EARTHQUAKE = settings["pivot"] == "earthquake"
USE_OTHER_ACCELERATION = settings["pivot"] == "other"
REAL_EARTHQUAKE = settings["earthquake"] == "real"
earthquake_specific = Path(settings["earthquake_data_folder"]) / settings["earthquake_specific"]
USE_DAMPING = settings["damping"] != 0 
DAMPING = settings["damping"]

FOLLOW_PIVOT = settings["follow_pivot"]
TRACE_PATH = settings["trace_path"]
PHASE = settings["run_phase_space"]
GIF_FRAMERATE = settings["gif_framerate"]
SAVE_IMAGES = settings["save_images"]
SAVE_GIF = settings["gif"] and SAVE_IMAGES
RUN_ALL = len(settings["methods"])>1 # runs all integration techniques

START, END, FRAMERATE, PLAY_BACK = settings["start"], settings["end"], settings["framerate"], settings["playback"]
RUNTIME = END - START
OUTPUT_FOLDER = Path(settings["output"])

G = settings["g"]
RTOL, ATOL = settings["rtol"], settings["atol"]

PLOT_SHOW = settings["plot_show"]
if not PLOT_SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pendulum_library as pendulum

#---------------------more validation-------------------

for arr in (
    masses, lengths, a_cm, b_cm,
    I_cm, initial_thetas, initial_omegas
):
    if np.shape(arr) != (N,):
        raise ValueError("All per-link arrays must have length N.")

if np.any(masses <= 0) or np.any(lengths <= 0) or np.any(I_cm < 0):
    raise ValueError("Invalid masses, lengths, or inertias.")

# ------------------- Numerical Integration -------------------

t = np.linspace(START, END, int(RUNTIME*FRAMERATE))

y0 = np.column_stack((
    initial_thetas,
    initial_omegas
)).ravel()


solutions = {}
solver_runtimes = {}

motion_args = (
    masses, I_cm, lengths, a_cm, b_cm,
    initial_thetas, ORDER, G,
    USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
    REAL_EARTHQUAKE, earthquake_specific,
    USE_DAMPING, DAMPING
)

for m in settings["methods"]:
    start_time = time.perf_counter()
    solutions[m] = solve_ivp(
        pendulum.motion, (START, END), y0, t_eval=t,
        method=m, rtol=RTOL, atol=ATOL, args=motion_args
    )
    solver_runtimes[m] = time.perf_counter() - start_time

# ------------------- High-Accuracy Reference -------------------

if RUN_ALL:
    reference = solve_ivp(
        pendulum.motion, (START, END), y0, t_eval=t,
        method="DOP853", rtol=1e-12, atol=1e-14,
        args=motion_args
    )
    if not reference.success:
        raise RuntimeError(f"Reference integration failed: {reference.message}")
    theta_reference = reference.y[0::2]

# ------------------- Creating Plots -------------------

angular_errors = []
function_evaluations = []

errors = []

for method, solution in solutions.items():
    print(f"Processing {method}")

    if not solution.success:
        print(f"{method} failed: {solution.message}")
        angular_errors.append(np.nan)
        errors.append(np.nan)
        function_evaluations.append(solution.nfev)
        continue

    method_folder = OUTPUT_FOLDER / method
    method_folder.mkdir(parents=True, exist_ok=True)

    function_evaluations.append(solution.nfev)

    if RUN_ALL:
        angle_difference = np.arctan2(
            np.sin(solution.y[0::2] - theta_reference),
            np.cos(solution.y[0::2] - theta_reference)
        )
        mean_squared_error = np.mean(angle_difference**2, axis=0)
        angular_errors.append(np.sqrt(
            np.trapezoid(mean_squared_error, x=solution.t)
            / (solution.t[-1] - solution.t[0])
        ))

    fig, fig_angles, fig_energy, fig_x, fig_angular_x, animation, fig_phase, animation_phase = pendulum.solve(
        solution, masses, I_cm, lengths, a_cm, b_cm,
        G, ORDER,
        USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
        REAL_EARTHQUAKE, earthquake_specific,
        USE_DAMPING, DAMPING,
        FOLLOW_PIVOT, TRACE_PATH, PHASE,
        FRAMERATE, GIF_FRAMERATE, PLAY_BACK,
        errors, motion_args
    )

    if SAVE_IMAGES:
        # Save all static plots
        fig_angles.savefig(
            method_folder / "angles.png",
            dpi=300, bbox_inches="tight"
        )

        fig_energy.savefig(
            method_folder / "energy.png",
            dpi=300, bbox_inches="tight"
        )

        fig_x.savefig(
            method_folder / "x_position.png",
            dpi=300, bbox_inches="tight"
        )

        fig_angular_x.savefig(
            method_folder / "x_angular_acceleration.png",
            dpi=300, bbox_inches="tight"
        )

    if SAVE_GIF:
        # Save pendulum animation as GIF
        print(f"Saving {method} GIF...", flush=True)
        animation.save(
            method_folder / "pendulum.gif",
            writer="pillow",
            fps=FRAMERATE * PLAY_BACK / max(1, round(FRAMERATE / GIF_FRAMERATE)),
            dpi=80
        )
        
        # Save pendulum phase space animation as GIF
        print(f"Saving {method} Phase Space GIF...", flush=True)
        animation_phase.save(
            method_folder / "pendulum_phase.gif",
            writer="pillow",
            fps=FRAMERATE * PLAY_BACK / max(1, round(FRAMERATE / GIF_FRAMERATE)),
            dpi=80
        )

        print(f"Finished {method} GIFs.", flush=True)
    if PLOT_SHOW:
        plt.show()

    plt.close('all')


# ------------------- Solver Accuracy Comparison -------------------
if RUN_ALL: pendulum.solver_accuracy(settings["methods"], angular_errors, errors, solver_runtimes, function_evaluations, OUTPUT_FOLDER)

print("Finished all simulations.")
if SAVE_IMAGES:
    print(f"All output saved to {OUTPUT_FOLDER.resolve()}")

if PLOT_SHOW:
    plt.show()
plt.close('all')