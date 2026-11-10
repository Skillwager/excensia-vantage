/** A small seeded random source (mulberry32), so the simulated match is the same every time. */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** 0 <= value < 1 */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  int(min: number, maxInclusive: number): number {
    return Math.floor(this.range(min, maxInclusive + 1));
  }

  chance(probability: number): boolean {
    return this.next() < probability;
  }

  /** Standard normal (Box–Muller). */
  normal(): number {
    const u = Math.max(1e-12, this.next());
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Log-normal with the given median, clamped. A long right tail. */
  logNormal(median: number, sigma: number, min: number, max: number): number {
    const value = median * Math.exp(sigma * this.normal());
    return value < min ? min : value > max ? max : value;
  }

  /** Index picked in proportion to the weights. */
  pick(weights: readonly number[]): number {
    let total = 0;
    for (const w of weights) total += w;
    let at = this.next() * total;
    for (let i = 0; i < weights.length; i++) {
      at -= weights[i];
      if (at < 0) return i;
    }
    return weights.length - 1;
  }

  sign(): 1 | -1 {
    return this.next() < 0.5 ? -1 : 1;
  }
}
