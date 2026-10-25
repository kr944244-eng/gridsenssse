/* ============ GridSense Assistant (chat widget) ============ */
let chatStarted = false, chatBusy = false;

function toggleSidebar(open){ document.body.classList.toggle("side-open", open); }

function openChat(){
  document.body.classList.add("chat-open");
  toggleSidebar(false);
  if (!chatStarted){
    chatStarted = true;
    addMsg("bot", "Hi! 👋 I'm the **GridSense Assistant**. Ask me about your latest result, how to make the grid stable, "
      + "or try a what-if like *\"what if industrial reaction time is 2?\"*");
    setChips(["What is my last result?", "Why is my grid unstable?", "How do I make it stable?", "How accurate is the model?"]);
  }
  setTimeout(() => document.getElementById("chatText").focus(), 150);
}
function closeChat(){ document.body.classList.remove("chat-open"); }
document.addEventListener("keydown", e => { if (e.key === "Escape") closeChat(); });

/* minimal, safe markdown: escape first, then **bold**, _italic_, `code`, lists */
function md(text){
  const e = String(text).replace(/[&<>]/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;"})[c]);
  const inline = s => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[\s(])_(.+?)_(?=[\s.,)!?]|$)/g, "$1<i>$2</i>")
                       .replace(/\*(.+?)\*/g, "<i>$1</i>").replace(/`(.+?)`/g, "<code>$1</code>");
  let html = "", list = null;
  e.split("\n").forEach(line => {
    const m = line.match(/^\s*(?:-|\d+\.)\s+(.*)$/);
    const cont = line.match(/^\s{2,}(\S.*)$/);
    if (m){ if (!list){ html += "<ul>"; list = true; } html += "<li>" + inline(m[1]) + "</li>"; }
    else if (cont && list){ html = html.replace(/<\/li>$/, "<br><span class='sub'>" + inline(cont[1]) + "</span></li>"); }
    else { if (list){ html += "</ul>"; list = null; } html += line.trim() ? "<p>" + inline(line) + "</p>" : ""; }
  });
  if (list) html += "</ul>";
  return html;
}

function addMsg(who, text, link){
  const body = document.getElementById("chatBody");
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.innerHTML = who === "bot" ? md(text) : String(text).replace(/[&<>]/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;"})[c]);
  if (link){
    const a = document.createElement("a");
    a.className = "btn sm"; a.href = link.href; a.textContent = link.label; a.style.marginTop = "8px";
    div.appendChild(a);
  }
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  return div;
}

function setChips(chips){
  const box = document.getElementById("chatChips");
  box.innerHTML = "";
  (chips || []).slice(0, 4).forEach(c => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = c;
    b.onclick = () => ask(c);
    box.appendChild(b);
  });
}

async function ask(text){
  if (chatBusy || !text.trim()) return;
  chatBusy = true;
  addMsg("me", text);
  setChips([]);
  const typing = addMsg("bot typing", "");
  typing.innerHTML = "<span></span><span></span><span></span>";
  try {
    const r = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({message:text})});
    if (r.status === 401){ location.href = "/login"; return; }
    const data = await r.json();
    typing.remove();
    addMsg("bot", data.reply || "Sorry, I didn't get that.", data.link);
    setChips(data.chips);
  } catch (e){
    typing.remove();
    addMsg("bot", "I couldn't reach the server. Check that the website is still running and try again.");
  } finally { chatBusy = false; }
}

function sendChat(ev){
  ev.preventDefault();
  const inp = document.getElementById("chatText");
  const t = inp.value; inp.value = "";
  ask(t);
}
