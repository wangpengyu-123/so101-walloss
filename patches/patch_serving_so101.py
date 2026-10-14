"""Patch wall-x serving to accept SO-ARM101 6-dim joint state (so101_joint_pos)."""
p1 = "/root/autodl-tmp/wallx_ft/wall-x/wall_x/_vendor/harrix/serving/_wallx_infer/base_dataclass.py"
s = open(p1).read()
old = '    "left_joint": 6,'
new = '    "left_joint": 6,\n    "so101_joint_pos": 6,'
assert old in s, "dof_dims left_joint not found"
s = s.replace(old, new, 1)
old2 = '                "state_right_joint": None,'
new2 = '                "state_right_joint": None,\n                "state_so101_joint_pos": None,'
assert old2 in s, "data init state_right_joint not found"
s = s.replace(old2, new2, 1)
open(p1, "w").write(s)
print("base_dataclass.py patched")

p2 = "/root/autodl-tmp/wallx_ft/wall-x/wall_x/_vendor/harrix/serving/_wallx_infer/robot.py"
s2 = open(p2).read()
old3 = '''    def get_observation(self, state, views):
        robot_state_action_data = RobotStateActionData(config=self.config)
        if "follow1_pos" in state or "follow2_pos" in state:
            ingest_websocket_follow_pos(state, self.config, robot_state_action_data)'''
new3 = '''    def get_observation(self, state, views):
        robot_state_action_data = RobotStateActionData(config=self.config)
        if "so101_joint_pos" in state:
            robot_state_action_data.save_state_data_with_key(
                np.asarray(state["so101_joint_pos"]), "so101_joint_pos"
            )
            state = dict(state)
            state.pop("follow1_pos", None)
            state.pop("follow2_pos", None)
        if "follow1_pos" in state or "follow2_pos" in state:
            ingest_websocket_follow_pos(state, self.config, robot_state_action_data)'''
assert old3 in s2, "get_observation pattern not found"
s2 = s2.replace(old3, new3, 1)
open(p2, "w").write(s2)
print("robot.py patched")
