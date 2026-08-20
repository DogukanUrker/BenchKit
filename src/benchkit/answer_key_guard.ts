import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync, realpathSync } from "node:fs";
import { isAbsolute, normalize, resolve } from "node:path";

const AUDIT_LOG = "/tmp/benchkit-answer-key-guard.jsonl";
const ANSWER_COMPONENT = /(^|[\\/])\.(meta|approaches)([\\/]|$)/i;
const ANSWER_NAME = /(^|[\\/])[^\\/]*(example|reference|proof)[^\\/]*([\\/]|$)/i;

function forbidden(value: unknown, cwd: string): boolean {
  if (typeof value !== "string" || !value.trim()) return false;
  const candidate = value.trim();
  const normalized = normalize(isAbsolute(candidate) ? candidate : resolve(cwd, candidate));
  let resolved = normalized;
  try { resolved = realpathSync(normalized); } catch { /* missing/glob path */ }
  return ANSWER_COMPONENT.test(candidate) || ANSWER_NAME.test(candidate)
    || ANSWER_COMPONENT.test(normalized) || ANSWER_NAME.test(normalized)
    || ANSWER_COMPONENT.test(resolved) || ANSWER_NAME.test(resolved);
}

function inspectedValues(toolName: string, input: Record<string, unknown>): string[] {
  if (toolName === "bash") return [String(input.command ?? "")];
  if (["read", "grep", "find", "ls"].includes(toolName)) {
    return ["path", "pattern", "glob", "query"]
      .map((key) => input[key])
      .filter((value): value is string => typeof value === "string");
  }
  return [];
}

export default function answerKeyGuard(pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    const input = event.input as Record<string, unknown>;
    const matches = inspectedValues(event.toolName, input).filter((value) =>
      forbidden(value, ctx.cwd),
    );
    if (!matches.length) return undefined;

    appendFileSync(AUDIT_LOG, JSON.stringify({
      tool: event.toolName,
      arguments: input,
      matches,
      cwd: ctx.cwd,
    }) + "\n", { encoding: "utf8", mode: 0o600 });
    return {
      block: true,
      reason: "BenchKit denied access to an answer-key path",
    };
  });
}
