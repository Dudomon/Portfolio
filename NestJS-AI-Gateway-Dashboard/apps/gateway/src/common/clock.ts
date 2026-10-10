/** Seconds since the epoch, the unit OpenAI uses for `created`. */
export function unixSeconds(): number {
  return Math.floor(Date.now() / 1000);
}
