# TowerFall RL

Teaching an agent to play [TowerFall Ascension](http://www.towerfall-game.com) with reinforcement
learning.

> **Work in progress.** This is an experimental project. The first task trains, the bigger goals
> (versus matches and self-play) are not built yet, and a lot of ideas will be tried, kept or
> thrown away along the way. Expect the code, the rewards and the results to change.

## Status

Working today:

- A Python client that launches the game, configures a match and plays it frame by frame. Several
  copies of the game run in parallel for training.
- Our own fork of the TowerfallAi mod (in `mod/`), which is what lets a program read the game
  state and press buttons. See [The mod](#the-mod) for what the fork adds.
- A first training task, the **slime hunt**: one archer with three arrows on a flat floor against
  three slimes, trained with PPO.

Planned, roughly in this order:

- A 1v1 versus environment, with rewards taken from the mod's kill and round events.
- Self-play: the agent trains against earlier versions of itself.
- A richer observation that uses what the fork exposes: whether the opponent is drawing the bow,
  dodge cooldowns, and which arrows in the air are dangerous and whose they are.
- Faster training by holding the buttons inside the mod instead of asking the agent every frame.
- Whatever else turns out to be worth trying: other reward designs, network architectures and
  algorithms.

## How it works

```
TowerFall.exe + mod  <--  JSON over a local socket  -->  Python client  -->  Gymnasium env  -->  PPO
```

The mod runs inside the game and reads its objects directly, so the agent never looks at pixels.
Every frame the game sends the list of entities (archers, arrows, enemies, items) with their
positions, speeds and states, then waits for the agent's buttons before it computes the next
frame. Because the game waits, the agent never misses a frame, and with the frame rate cap removed
the game runs many times faster than real time.

## Requirements

- Windows, TowerFall Ascension on Steam, Steam running
- [uv](https://docs.astral.sh/uv/)
- The .NET SDK, to build the mod: `winget install Microsoft.DotNet.SDK.10`
- An NVIDIA GPU. PyTorch is pinned to the CUDA 12.8 build, which RTX 50-series cards need.

## Setup

1. Clone with the mod, which is a git submodule:

   ```
   git clone --recurse-submodules https://github.com/absor97/towerfall-ascension-reinforcement-learning-ai.git
   ```

2. Install the Python environment:

   ```
   uv sync
   ```

3. Install the mod (once, and again after every game update, which removes it):

   - Download `TowerfallAi_v0_1_1.zip` from the
     [upstream mod's releases](https://github.com/TowerfallAi/towerfall-ai/releases) and unzip it
     into the game folder, next to `TowerFall.exe`. This provides `Patcher.exe` and the libraries.
   - Close the game and run `Patcher.exe` from that folder. It rewrites `TowerFall.exe` and keeps
     the original in `Patcher/Original`. To reinstall after an update, delete the `Patcher` folder
     first.
   - Build our version of the mod and patch it in:

     ```
     mod\install.ps1
     ```

     Run it again after every change to the C# code in `mod/`. The Python client refuses to talk
     to another version of the mod.

4. With the game closed, set these options in `tf_saveData` in the game folder:

   ```xml
   <ReplayMode>Off</ReplayMode>
   <Fullscreen>false</Fullscreen>
   ```

   The end-of-round replay crashes the game when it runs faster than real time, and training
   minimizes the game windows, which needs windowed mode.

5. Create `steam_appid.txt` in the game folder containing only the app id:

   ```
   251470
   ```

   Without it the game asks Steam to launch it, and Steam refuses to start a second copy.
   Training runs one copy of the game per parallel environment.

The game folder defaults to `C:\Program Files (x86)\Steam\steamapps\common\TowerFall`. Set
`TOWERFALL_PATH` if yours is elsewhere. To remove the mod, use "Verify integrity of game files"
in Steam.

## Check that it works

```
uv run scripts/random_agents.py --frames 3000 --fps 0
```

This launches the game with `--aimod`, plays a 1v1 between two random agents and prints the
simulation speed. `--fps 60` runs at normal speed for watching; `--kill` closes the game after.

```
uv run pytest
```

The tests run against fake game data and do not need TowerFall.

## Training

The slime hunt is the only task so far. Random play wins about 44% of episodes. The rewards are
defined by the `Rewards` class in `src/towerfall_rl/envs/slime_hunt.py`: by default kills, a bonus
for stomping, a penalty for dying, and small terms for keeping an arrow and dodging near a slime.
Pass `--rewards simple` to train on kills, death and time only.

```
uv run scripts/train.py --timesteps 1000000 --envs 8
uv run tensorboard --logdir runs
```

This starts 8 minimized copies of the game, each on its own CPU core, and trains PPO at about
1400 to 1500 agent steps per second on a 12-core Ryzen 9 9900X (one agent step is 4 game frames).
Watch `rollout/ep_won_mean` in TensorBoard. The model and checkpoints are saved under
`runs/<name>/`.

```
uv run scripts/play.py runs/<name>/model.zip                         # watch at normal speed
uv run scripts/play.py runs/<name>/model.zip --fps 0 --episodes 300  # measure the win rate
```

## The mod

`mod/` is our fork of [TowerfallAi/towerfall-ai](https://github.com/TowerfallAi/towerfall-ai), kept
in [its own repository](https://github.com/absor97/towerfall-ai-rl). Compared to the upstream
release it adds what reinforcement learning needs:

- the real speed of archers and arrows instead of the change in position
- for archers the dodge cooldown and whether they are aiming; for arrows who shot them and
  whether they are dangerous
- events: every death with its killer and cause, and the scores at the end of a round
- seeded resets, so the same seed and the same buttons replay the same episode
- holding the buttons for several frames without an update in between
- a switch to turn drawing off

`mod/README.md` documents every message the mod sends and accepts.

## Layout

```
mod/                       our fork of the mod (C#, a git submodule)
src/towerfall_rl/client/   socket client for the mod: launch, configure, join, step
src/towerfall_rl/envs/     Gymnasium environments, observation and action encoding
scripts/                   runnable entry points
tests/                     tests against fake game data, no TowerFall needed
```

## Credits

- [TowerFall Ascension](http://www.towerfall-game.com) is made by Extremely OK Games. You need to
  own the game; nothing of it is included here. This project is not affiliated with its makers.
- The mod is based on [TowerfallAi/towerfall-ai](https://github.com/TowerfallAi/towerfall-ai).

## License

The code in this repository is under the MIT license, see `LICENSE`. The mod in `mod/` is a
separate repository and is not covered by it.
