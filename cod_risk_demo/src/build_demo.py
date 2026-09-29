"""Step 8: build the offline demo page demo/index.html.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

The page is one self-contained HTML file: the contents of reports/demo_orders.json
are embedded in it, so it works offline with no server and no data calls.
"""
import json

from src.common import REPORTS_DIR, ROOT, write_text

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COD Risk Score demo</title>
<style>
  :root {
    --bg: #f5f6f8; --card: #ffffff; --ink: #1d2330; --muted: #5d6677; --line: #e3e6ec;
    --brand: #ee4d2d; --green: #1e8e3e; --amber: #c77700; --red: #d93025;
    --green-bg: #e6f4ea; --amber-bg: #fff4e0; --red-bg: #fce8e6;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font-family: "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif; font-size: 20px; }
  .wrap { max-width: 1760px; margin: 0 auto; padding: 32px 40px 24px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
  h1 { font-size: 38px; margin: 0; letter-spacing: -0.5px; }
  h1 span { color: var(--brand); }
  .pick { display: flex; gap: 16px; }
  .pick button { font-size: 24px; font-weight: 600; padding: 16px 36px; border-radius: 14px; cursor: pointer;
                 border: 3px solid var(--ink); background: var(--card); color: var(--ink); }
  .pick button.on { background: var(--ink); color: #fff; }
  .grid { display: grid; grid-template-columns: 1fr 440px; gap: 32px; margin-top: 28px; align-items: start; }
  .card { background: var(--card); border-radius: 20px; padding: 28px 32px; box-shadow: 0 2px 10px rgba(20,30,50,.06); }
  .card h2 { font-size: 22px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 0 0 16px; }
  .top { display: grid; grid-template-columns: 1.1fr 1fr; gap: 28px; }
  .facts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 24px; }
  .fact { border-bottom: 1px solid var(--line); padding: 8px 0; }
  .fact .k { font-size: 16px; color: var(--muted); }
  .fact .v { font-size: 26px; font-weight: 650; }
  .score { display: flex; flex-direction: column; align-items: flex-start; gap: 16px; }
  .risk { font-size: 96px; font-weight: 800; line-height: 1; letter-spacing: -2px; }
  .risk small { display: block; font-size: 20px; font-weight: 500; color: var(--muted); letter-spacing: 0; margin-top: 8px; }
  .badge { font-size: 28px; font-weight: 750; padding: 10px 24px; border-radius: 999px; }
  .t0 { background: var(--green-bg); color: var(--green); }
  .t1 { background: var(--amber-bg); color: var(--amber); }
  .t2 { background: var(--red-bg); color: var(--red); }
  .action { font-size: 24px; font-weight: 600; }
  .action b { color: var(--muted); font-weight: 500; }
  .reasons { margin-top: 28px; }
  .reason { display: grid; grid-template-columns: 56px 1fr 260px; align-items: center; gap: 16px; padding: 14px 0;
            border-bottom: 1px solid var(--line); }
  .reason:last-child { border-bottom: 0; }
  .arrow { font-size: 34px; font-weight: 800; text-align: center; }
  .up { color: var(--red); } .down { color: var(--green); }
  .rtext { font-size: 26px; font-weight: 600; }
  .rtext small { display: block; font-size: 17px; color: var(--muted); font-weight: 500; }
  .bar { height: 18px; border-radius: 9px; background: var(--line); position: relative; overflow: hidden; }
  .bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 9px; }
  .bar i.up { background: var(--red); } .bar i.down { background: var(--green); }
  /* phone */
  .phone { width: 420px; height: 820px; border-radius: 52px; background: #111; padding: 16px; margin: 0 auto;
           box-shadow: 0 20px 40px rgba(0,0,0,.18); }
  .screen { background: #fff; border-radius: 40px; height: 100%; overflow: hidden; display: flex; flex-direction: column; }
  .notch { height: 34px; display: flex; justify-content: center; align-items: center; }
  .notch i { width: 120px; height: 22px; background: #111; border-radius: 12px; }
  .appbar { background: var(--brand); color: #fff; font-size: 22px; font-weight: 700; padding: 16px 22px; }
  .pbody { padding: 22px; display: flex; flex-direction: column; gap: 14px; flex: 1; }
  .ptitle { font-size: 28px; font-weight: 750; line-height: 1.2; }
  .psub { font-size: 18px; color: var(--muted); line-height: 1.35; }
  .check { width: 84px; height: 84px; border-radius: 50%; background: var(--green-bg); color: var(--green);
           font-size: 48px; display: flex; align-items: center; justify-content: center; margin: 24px auto 4px; }
  .pbtn { font-size: 21px; font-weight: 650; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid var(--line); }
  .pbtn.main { background: var(--brand); color: #fff; border-color: var(--brand); }
  .pbtn.alt { color: var(--ink); }
  .pbtn.offer { background: var(--amber-bg); border-color: #f3d19c; color: #7a4a00; text-align: left; }
  .pbtn.offer small { display: block; font-size: 15px; font-weight: 500; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .phone-cap { text-align: center; color: var(--muted); font-size: 17px; margin-top: 14px; }
  footer { margin-top: 26px; text-align: center; color: var(--muted); font-size: 18px; }
  @media (max-width: 1200px) { .grid { grid-template-columns: 1fr; } .top { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span>COD Risk Score</span> &middot; checkout demo</h1>
    <div class="pick">
      <button id="btnA" onclick="show('A')">Order A</button>
      <button id="btnB" onclick="show('B')">Order B</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="top">
        <div>
          <h2>Order facts</h2>
          <div class="facts" id="facts"></div>
        </div>
        <div class="score">
          <h2>Risk of failed delivery</h2>
          <div class="risk" id="risk"></div>
          <div class="badge" id="badge"></div>
          <div class="action" id="action"></div>
        </div>
      </div>
      <div class="reasons">
        <h2>Top 3 reasons</h2>
        <div id="reasons"></div>
      </div>
    </div>

    <div>
      <div class="phone"><div class="screen">
        <div class="notch"><i></i></div>
        <div class="appbar">Checkout</div>
        <div class="pbody" id="phone"></div>
      </div></div>
      <div class="phone-cap">What the buyer sees</div>
    </div>
  </div>

  <footer>Synthetic data, for illustration only. Not Shopee data. &middot; Model: <span id="model"></span></footer>
</div>

<script>
const DATA = __DATA__;
const TIER_NAME = {0: "Tier 0 · Low risk", 1: "Tier 1 · Confirm", 2: "Tier 2 · High risk"};

function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }

function phone(o) {
  const value = o.facts["Order value"];
  if (o.tier === 0) {
    return `<div class="check">&#10003;</div>
      <div class="ptitle" style="text-align:center">Order placed</div>
      <div class="psub" style="text-align:center">${value} &middot; Cash on delivery.<br>Pay the courier when your parcel arrives.</div>`;
  }
  let html = `<div class="ptitle">Ready to ship ${value} COD</div>
    <div class="psub">Please confirm you will be home to receive and pay for this parcel.</div>
    <div class="pbtn main">Confirm</div>
    <div class="row2"><div class="pbtn alt">Cancel</div><div class="pbtn alt">Pay now</div></div>`;
  if (o.tier === 2) {
    html += `<div class="pbtn offer">Pay now and get &#3647;10 coins<small>Faster delivery, no cash needed</small></div>
      <div class="pbtn offer">Keep COD with &#3647;20 deposit<small>Deposit is returned on delivery</small></div>`;
  }
  return html;
}

function show(key) {
  const o = DATA.orders[key];
  document.getElementById("btnA").classList.toggle("on", key === "A");
  document.getElementById("btnB").classList.toggle("on", key === "B");

  const facts = document.getElementById("facts"); facts.innerHTML = "";
  for (const [k, v] of Object.entries(o.facts)) {
    const f = el("div", "fact"); f.append(el("div", "k", k), el("div", "v", v)); facts.append(f);
  }
  const risk = document.getElementById("risk");
  risk.innerHTML = `${(o.risk * 100).toFixed(1)}%<small>chance this COD parcel fails (average is about 2.6%)</small>`;
  risk.style.color = ["var(--green)", "var(--amber)", "var(--red)"][o.tier];
  const badge = document.getElementById("badge");
  badge.className = "badge t" + o.tier; badge.textContent = TIER_NAME[o.tier];
  document.getElementById("action").innerHTML = `<b>Action:</b> ${o.action}`;

  const maxAbs = Math.max(...o.reasons.map(r => Math.abs(r.shap)));
  const box = document.getElementById("reasons"); box.innerHTML = "";
  for (const r of o.reasons) {
    const row = el("div", "reason");
    row.append(el("div", "arrow " + r.direction, r.direction === "up" ? "&#9650;" : "&#9660;"));
    row.append(el("div", "rtext", `${r.text}<small>${r.direction === "up" ? "Pushes risk up" : "Pushes risk down"} (${r.name})</small>`));
    const bar = el("div", "bar"); const fill = el("i", r.direction);
    fill.style.width = (100 * Math.abs(r.shap) / maxAbs).toFixed(0) + "%"; bar.append(fill); row.append(bar);
    box.append(row);
  }
  document.getElementById("phone").innerHTML = phone(o);
}

document.getElementById("model").textContent = DATA.model;
show(location.hash === "#B" ? "B" : "A");
</script>
</body>
</html>
"""


def main():
    data = json.loads((REPORTS_DIR / "demo_orders.json").read_text(encoding="utf-8"))
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # safe inside <script>
    write_text(ROOT / "demo" / "index.html", TEMPLATE.replace("__DATA__", embedded))
    print("Wrote demo/index.html (self-contained, works offline)")


if __name__ == "__main__":
    main()
