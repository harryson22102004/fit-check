const Persist = (() => {
  const KEY = "smartAttendance.v1";

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return { students: [], attendance: [], events: [] };
      const data = JSON.parse(raw);
      return {
        students: data.students || [],
        attendance: data.attendance || [],
        events: data.events || [],
      };
    } catch {
      return { students: [], attendance: [], events: [] };
    }
  }

  function save(data) {
    const payload = {
      students: data.students || [],
      attendance: data.attendance || [],
      events: data.events || [],
      saved_at: new Date().toISOString(),
    };
    try {
      localStorage.setItem(KEY, JSON.stringify(payload));
    } catch (err) {
      console.warn("Could not persist roster locally", err);
    }
  }

  async function syncFromServer() {
    const res = await fetch("/api/snapshot");
    if (!res.ok) return load();
    const snap = await res.json();
    save(snap);
    return snap;
  }

  function removeStudent(id) {
    const data = load();
    const sid = String(id).toUpperCase();
    data.students = data.students.filter((s) => s.student_id !== sid);
    data.attendance = data.attendance.filter((a) => a.student_id !== sid);
    data.events = data.events.filter((e) => (e.student_id || "") !== sid);
    save(data);
  }

  async function hydrate() {
    const local = load();
    let server;
    try {
      const res = await fetch("/api/snapshot");
      server = res.ok ? await res.json() : { students: [], attendance: [], events: [] };
    } catch {
      return;
    }
    const serverN = (server.students || []).length;
    const localN = local.students.length;
    const serverA = (server.attendance || []).length;
    const localA = local.attendance.length;

    if (localN > serverN || localA > serverA) {
      const restored = await fetch("/api/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(local),
      });
      if (restored.ok && !sessionStorage.getItem("sa-restored")) {
        sessionStorage.setItem("sa-restored", "1");
        location.reload();
      }
      return;
    }
    if (serverN || serverA) save(server);
  }

  return { load, save, syncFromServer, removeStudent, hydrate };
})();

document.addEventListener("DOMContentLoaded", () => {
  Persist.hydrate();
});
