#pragma once

class Session;

// 负责提示符、读取输入、/exit、/clear、终端退出处理及回答显示。
// 每轮调用 session.run_turn(input)，不直接操作编解码或生成器。
void run_repl(Session& session);
