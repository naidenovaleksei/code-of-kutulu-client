GAME: Code of Kutulu — compact rules for an agent

GOAL
- Survive longer than the other 3 explorers.
- You lose when your sanity reaches 0 before the others, or if you output an invalid command.
- If a command is syntactically valid but impossible, it becomes WAIT.

GRID / DISTANCE
- 4-player game on a rectangular grid.
- Coordinates: {0,0} is top-left.
- Default distance: Manhattan distance, unless stated otherwise.
- MOVE follows shortest walkable path.
- Wanderer targeting/movement uses shortest walkable path.
- “Grouped sanity” check ignores walls: another explorer within distance <= 2 counts as nearby even through walls.
- PLAN/LIGHT range does NOT penetrate walls.
- Slasher line-of-sight (LoS): same row/column with no wall in between.

MAP CELLS
- # = wall
- . = empty
- w = wanderer spawn point
- U = shelter

EXPLORER
- Initial sanity: 250
- Base passive sanity loss each turn:
  - lonely: sanityLossLonely (commonly 3–6 depending on league settings)
  - grouped: sanityLossGroup (commonly 1–3 depending on league settings)
- Grouped = at least one other explorer within Manhattan distance <= 2, walls ignored.

SPAWN / GLOBAL VARIABLE CONSTANTS
- These values are provided in input and may vary by game:
  - sanityLossLonely
  - sanityLossGroup
  - wandererSpawnTime
  - wandererLifeTime
- Welcome/update page explicitly says isolation and wanderer values can vary. :contentReference[oaicite:2]{index=2}
- Statement confirms the four runtime-modified constants above. :contentReference[oaicite:3]{index=3}

MAIN THREATS

1) WANDERERS
- Spawn from fixed spawn points.
- A new wanderer is spawned at the spawn point farthest from any player, ignoring walls.
- Spawning delay: wandererSpawnTime (statement examples mention 3–6 in this league).
- After spawn:
  1. Target nearest explorer by shortest walkable path.
  2. Move 1 cell toward target.
  3. Recompute each turn.
- If wanderer reaches a player cell:
  - all players on that cell are spooked
  - each loses 20 sanity
  - wanderer disappears
- Tie-breaking:
  - keep previous target if still tied
  - otherwise choose randomly
- If multiple shortest next steps exist, move priority is:
  - Up, Right, Down, Left
- Recall:
  - if a wanderer does not spook anyone within wandererLifeTime turns, it is removed. :contentReference[oaicite:4]{index=4}

2) SLASHERS
- New rule in this league. Slashers can appear and persist. :contentReference[oaicite:5]{index=5}
- A slasher starts spawning on the same cell as an explorer the first time that explorer drops below 200 sanity.
- Maximum slashers at once: 4 (one per explorer).
- Slashers never permanently disappear from the map.
- State machine:
  1. SPAWN: 6 turns
  2. WANDERING:
     - behaves like a wanderer
     - if any explorer enters LoS, switch to STALKING
  3. STALKING:
     - stay still for 2 turns
     - keep tracking target position while target remains in LoS
     - then switch to RUSHING
  4. RUSHING:
     - instantly rush to target if still in LoS
     - otherwise rush to last seen position
     - all players on landing cell are spooked
     - then switch to STUNNED
  5. STUNNED:
     - inactive for 6 turns
     - then return to WANDERING
- Important edge case:
  - if an explorer is already in LoS immediately after spawn, the slasher skips WANDERING and STALKING and jumps straight to RUSHING.
- Targeting details:
  - keeps last target until target leaves LoS
  - when target leaves LoS:
    - choose nearest target currently in LoS
    - if none, remember last seen position
  - if a new explorer enters LoS, it becomes the new target
  - if multiple enter LoS simultaneously:
    - prefer previous target
    - then nearest distance
    - if still unresolved, slasher becomes STUNNED. :contentReference[oaicite:6]{index=6}

PLAYER ACTIONS
Exactly one action per turn:
- WAIT
- MOVE x y
- PLAN
- LIGHT
- YELL

EFFECTS / ABILITIES

1) PLAN
- Uses remaining plan charges; initial total available per explorer: 2.
- Duration: 5 turns
- Range: 2
- Walls block it
- Effect moves with the caster
- Each turn during effect:
  - every player in range gains +3 sanity
  - caster gains an additional +3 sanity for each OTHER player reassured

2) LIGHT
- Uses remaining light charges; initial total available per explorer: 3.
- Duration: 3 turns
- Range: 5
- Walls block it
- Effect on wanderers only:
  - add +4 to the distance of each cell when wanderers evaluate closest explorer
  - effectively discourages wanderers from targeting players near the light
- Does not say it affects slashers

3) YELL
- New rule in this league. :contentReference[oaicite:7]{index=7}
- Range: 1
- Affected nearby explorers are locked for 2 turns
- During lock they cannot do any action
- Works only once per target player for the whole game
- Turn-order implication:
  - YELL is resolved before spawn/move/effects
  - any nearby non-YELL action is converted to WAIT for that turn

4) SHELTERS
- New rule in this league. :contentReference[oaicite:8]{index=8}
- Fixed map cells marked U
- If you stay on a shelter, it restores +5 sanity per turn
- You still suffer passive sanity loss normally
- Shelter energy:
  - starts at 10
  - loses 1 energy each time it reassures an explorer
  - at 0 energy it stops restoring sanity
  - refills from 0 to 10 every 50 turns

EFFECT STACKING
- A player cannot have two active effects at the same time.
- Relevant active effects are PLAN / LIGHT / shelter effect / yell effect as represented in entities. At minimum, statement explicitly says a player cannot have two active effects at the same time. :contentReference[oaicite:9]{index=9}

TURN ORDER (VERY IMPORTANT)
Each turn resolves in this order:
1. All players receive state and output an action.
2. YELL resolves first:
   - any non-YELL action from an explorer within distance 1 of a yeller is converted to WAIT.
3. New minions spawn.
4. Players move.
5. Effects are applied.
6. Minions move.
7. Minions spook players sharing their cell.
8. Players lose sanity; wanderer recall timers tick down.
9. Remove:
   - players with 0 sanity
   - wanderers that expired or already spooked
   - effects that expired. :contentReference[oaicite:10]{index=10}

IMPORTANT CONSEQUENCES FOR SIMULATION
- YELL happens before everything else.
- A yelled target effectively loses both the current turn and the next turn.
- Shelter healing and PLAN happen in the “effects are applied” phase, before minion movement/spooking.
- Passive sanity loss happens after minion movement/spooking and after effects.
- Wanderers can be redirected by LIGHT, but slashers follow LoS-based rules and are not described as being affected by LIGHT.
- Slashers are long-term hazards because they persist forever once created.

INPUT FORMAT

Initialization:
1. width
2. height
3. next `height` lines = map rows
4. one line with:
   - sanityLossLonely
   - sanityLossGroup
   - wandererSpawnTime
   - wandererLifeTime

Per turn:
- first line: entityCount
- then `entityCount` entities:
  - entityType: EXPLORER | WANDERER | SLASHER | EFFECT_PLAN | EFFECT_LIGHT | EFFECT_SHELTER | EFFECT_YELL
  - id
  - x y
  - param0
  - param1
  - param2

Entity semantics:
- EXPLORER:
  - param0 = sanity
  - param1 = remaining PLAN uses
  - param2 = remaining LIGHT uses
- WANDERER / SLASHER:
  - if spawning: param0 = turns before spawn
  - else:
    - wanderer param0 = turns before recall
    - slasher param0 = turns before next state change
  - param1 = state:
    - 0 SPAWN
    - 1 WANDERING
    - 2 RUSH PREP
    - 3 RUSH
    - 4 STUN
  - param2 = target explorer id, or -1 on spawn
- SHELTER:
  - param0 = remaining energy
  - param1 = -1
  - param2 = -1
- EFFECT_*:
  - id = -1
  - x,y = origin
  - param0 = turns before fade
  - param1 = explorer id that created the effect
  - param2:
    - EFFECT_YELL => id of yelled player
    - otherwise -1

INPUT NOTE
- The first EXPLORER entity is always you.
- Your explorer id is not guaranteed to be 0. :contentReference[oaicite:11]{index=11}

OUTPUT FORMAT
- One line each turn:
  - WAIT
  - MOVE x y
  - PLAN
  - LIGHT
  - YELL
- Optional debug/message text may follow the command. :contentReference[oaicite:12]{index=12}

CONSTRAINTS
- 10 <= width <= 24
- 10 <= height <= 20
- 0 <= spawn points <= 8
- 0 <= shelters <= 8
- First turn time limit: 1000 ms
- Subsequent turn time limit: 50 ms. :contentReference[oaicite:13]{index=13}
