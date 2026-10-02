# oMLX-Chatkomfort für die MaxToken-Webkonsole

Status: geplant; keine Implementierung begonnen.

## Objective

Die bestehende MaxToken-Webkonsole bekommt einen alltagstauglichen Textchat nach dem Funktionsvorbild von oMLX: gespeicherte und durchsuchbare Gespräche, sichere Markdown-/Code-/Formeldarstellung, Kopieren, Bearbeiten, Regenerieren, Antwortvarianten, Systemprompt-Profile, Kontextanzeige, Import/Export und wählbares Erscheinungsbild. Bereits vorhandenes Streaming, Abbrechen, Thinking-Ausgabe, Ladeanzeige und Antwortmetriken bleiben erhalten.

Ziel ist funktionale Übernahme, nicht das Kopieren des kompletten oMLX-Frontends oder seiner Marke. Console- und Logs-Ansichten bleiben funktional unverändert. Nicht Teil der Umsetzung sind Swift-Launcher, native App, Multi-Modell-Inferenz, Modellbrowser, serverseitige Gesprächssynchronisierung, autonome Tools oder Websuche.

Echter Modellwechsel im Chat sowie Bild-, Audio-, Video- und Office-/PDF-Verarbeitung werden ausdrücklich als Folgearbeiten mit Backend-Voraussetzungen abgegrenzt. Eine Modell-Auswahlliste mit scheinbar wechselnden Namen darf nicht als Ersatz für tatsächlichen Modellwechsel erscheinen.

## Settled Decisions and Assumptions

### Befund und Quellen

- Der Chat lebt derzeit in einer einzigen HTML-Datei; Verlauf und Abbruchcontroller sind globale Variablen. Nachrichten werden mit dem Systemprompt an die OpenAI-Chat-API geschickt. Quellen: python/maxtoken/server/console.html:731, :746, :770, :778.
- Antworten werden als Klartext gerendert; Thinking und Antworttext werden während des Streams getrennt gesammelt. Quellen: python/maxtoken/server/console.html:804, :816.
- Temperatur, Ausgabelimit und Thinking sind bewusst serverweite Standardwerte im Console-Tab. Die bestehenden Tests schützen diese Trennung. Quellen: python/maxtoken/server/console.html:322; tests/server/test_console.py:37.
- /console wird direkt aus dem Python-Paket ausgeliefert; Paketdaten enthalten bisher nur HTML. Neue lokale Assets müssen deshalb sowohl über HTTP als auch im installierten Paket funktionieren. Quellen: python/maxtoken/server/api_server.py:462; pyproject.toml:78.
- /v1/models gibt genau das konfigurierte Modell mit Kontextlänge und Reasoning-Fähigkeiten zurück. Eine Modell-Pool-Auswahl existiert nicht. Quelle: python/maxtoken/server/openai_api.py:142.
- Modellwechsel ist bereits ein Daemon-Stop/Start-Vorgang, kein bloßer Wechsel des model-Feldes im Chatrequest. Quelle: python/maxtoken/daemon/README.md, Abschnitte „Control it“ und „HTTP API“.
- Die gemeinsame Nachrichten-Normalisierung weist nicht-textuelle Inhalte zurück. Bild- und Audiofelder im Requestschema belegen keine Inferenzunterstützung. Quellen: python/maxtoken/server/generation.py:203; python/maxtoken/server/api_models.py:7.
- Ein genauer Tokenzähler und die Anwendung der serverseitigen Thinking-Vorgabe sind bereits vorhanden und sollen wiederverwendet werden. Quellen: python/maxtoken/server/generation.py:294, :327; python/maxtoken/server/anthropic_api.py:145.
- oMLX liefert das Funktionsvorbild: Verlauf in ../omlx/omlx/admin/templates/chat.html:5174; Regenerieren und Varianten in :6648; Bearbeiten in :6954; Kopieren in :6716; Suche in :7454; Profile in :1985 und :4998; Kontextanzeige in :1886; Export in :4571; Import in :2449; Themes in :2557; Markdown in :7142; Formeln in :7570.
- oMLX ist Apache-2.0-lizenziert; vorhandene Drittanbieterhinweise stehen bei uns bereits in NOTICE. Quellen: ../omlx/LICENSE:1; NOTICE:13.

### Festgelegte Umsetzung

1. Die bestehende Webkonsole und ihre same-origin API bleiben der Einstieg. Kein React, Electron, neues Frontend-Buildsystem oder Ersatz des Servers.
2. Nur der Chatcode wird aus der großen HTML-Datei in kleine native ES-Module ausgelagert: python/maxtoken/server/static/chat_state.mjs für testbare Daten- und Verlaufoperationen sowie static/chat.mjs für DOM, Speicherung, Netzwerk und Streaming. static/chat.css enthält die neue Chatdarstellung.
3. Browserlokale Speicherung über localStorage, mit versioniertem Format. Gespräche bleiben auf diesen Browser und diese Server-Origin begrenzt. Die Oberfläche erklärt das ausdrücklich. Kein stilles Löschen alter Chats bei vollem Speicher.
4. Pro Browser-Tab maximal ein aktiver Generierungsrequest. Andere gespeicherte Gespräche dürfen währenddessen angesehen werden; der laufende Request bleibt fest an seine Gesprächs- und Antwort-ID gebunden. Bearbeiten und Löschen des betroffenen Gesprächs verlangen vorherigen Abbruch. Stop wirkt auf den laufenden Request, nicht auf den gerade sichtbaren Chat.
5. Serverweite Sampling- und Thinking-Standards bleiben im Console-Tab. Eine Aktion „Kreativer erneut versuchen“ darf nur für diesen einen Request die Temperatur überschreiben; sie verändert keine globale Konfiguration.
6. Systemprompt-Profile sind browserlokale Textvorlagen, keine serverseitigen Modellprofile. Jedes Gespräch speichert seinen eigenen Prompttext als Kopie, damit spätere Profiländerungen alte Gespräche nicht unbemerkt verändern.
7. Markdown, Sanitizer, Syntaxhervorhebung und Formeldarstellung verwenden etablierte lokale Bibliotheken statt Eigenbau. Nur die nötigen Distributionsdateien von Marked, DOMPurify, highlight.js und KaTeX werden unter static/vendor aufgenommen, mit Versionen und Lizenzhinweisen. Kein CDN, keine Nutzung der oMLX-Assets allein aufgrund ihrer Dateinamen ohne Prüfung der jeweiligen Lizenz.
8. Browserlokale Daten und Darstellung dürfen niemals unkontrolliert in API-Nachrichten gelangen. Nur der ausgewählte Gesprächspfad, unterstützte Rollen und Textinhalte werden übertragen; UI-Metadaten, nicht ausgewählte Varianten und Fehlermeldungen bleiben lokal.
9. Tests für reine JavaScript-Logik nutzen Nodes eingebauten Testrunner; keine neue Produktionsabhängigkeit. Reale Browserprüfung ergänzt die Python- und JavaScript-Tests. Ein Browser-Automatisierungsframework wird in diesem Plan nicht neu eingeführt.

## Implementation Plan

### Phase 1 — Grundlage und gespeicherte Gespräche

- [ ] **1. Chat modularisieren und Assets korrekt ausliefern.** Dateien: python/maxtoken/server/console.html, python/maxtoken/server/static/chat.mjs, python/maxtoken/server/static/chat_state.mjs, python/maxtoken/server/static/chat.css, python/maxtoken/server/api_server.py, pyproject.toml, tests/server/test_console.py. Nur Chatdarstellung und Chatrequest-Logik aus dem Inline-Skript lösen; vorhandene Console-/Logs-Polls, Hash-Navigation und API-Kopierfelder erhalten. Einen statischen Pfad ausschließlich für das Paket-Assetverzeichnis registrieren; alle Module, CSS-, Vendor- und Font-Dateien explizit in die Paketdaten aufnehmen. Beide Stop-Buttons an denselben Chatrequest anbinden. Bestehende Tests auf ausgelagerte Assets umstellen, ohne deren fachliche Schutzbedingungen aufzuweichen. Abnahme: installierte Konsole lädt ohne Netzwerkzugriff auf externe Hosts; Streaming, Uhr, Thinking, Metriken und Stop funktionieren weiterhin.

- [ ] **2. Persistentes Gesprächsmodell und sichere Speicherung implementieren.** Dateien: static/chat_state.mjs, static/chat.mjs; neue tests/browser/chat_state.test.mjs. Das Speicherformat bekommt eine Schema-Version, stabile Gesprächs-/Turn-/Antwort-IDs, Titel, Erstellungs-/Änderungszeit, Promptkopie, Entwurf, tatsächlich verwendetes Modell je Antwort, Antworttext, separat gespeichertes Reasoning, Usage, Abschlussstatus und ausgewählte Variante. Speicheroperationen kapseln, Schreibvorgänge während des Streams begrenzen und bei Abschluss oder Abbruch sofort sichern. Nach Seitenneuladen unfertige Antworten als unterbrochen markieren, nicht als weiterlaufend. Bei beschädigtem Format die Originaldaten unverändert erhalten und Download/Reset anbieten; bei unbekannter neuer Schema-Version nicht überschreiben. Bei Quotenfehlern sichtbare Warnung, In-Memory-Weiternutzung und Export anbieten, ohne stilles Entfernen von Gesprächen. storage-Ereignisse anderer Tabs erkennen und konkurrierende Bearbeitung desselben Gesprächs nicht unbemerkt überschreiben: lokale Fassung bei Konflikt als gesondertes Gespräch erhalten. Abnahme: Reload erhält Verlauf, Entwürfe und Prompt; beschädigte Daten und Speichergrenzen führen nicht zu stillem Datenverlust.

- [ ] **3. Gesprächsseitenleiste und Suche hinzufügen.** Dateien: console.html, static/chat.css, static/chat.mjs, static/chat_state.mjs, tests/browser/chat_state.test.mjs. Neue Chats, Gesprächsauswahl, Umbenennen, bestätigtes Löschen und Suche über Titel sowie sichtbare Nachrichten einführen; automatische Titel lokal aus dem ersten Nutzertext erzeugen, ohne zusätzliche Modellrequests. Nach Aktivität sortieren und eine klare Leeransicht anbieten. Auf kleinen Displays die Seitenleiste ein-/ausklappbar machen; Fokus und Auswahlzustand zugänglich kennzeichnen. Beim Gesprächswechsel Entwurf und Systemprompt aus dem jeweiligen Gespräch laden. Laufende Antworten bleiben dem ursprünglichen Gespräch zugeordnet. Abnahme: Suche findet Inhalte und Titel; Auswahl, Rename und Delete ändern nur das gewählte Gespräch und bleiben nach Reload erhalten.

### Phase 2 — Darstellung und Nachrichtenaktionen

- [ ] **4. Sichere Rich-Text-Darstellung und Kopieren integrieren.** Dateien: static/chat.mjs, static/chat.css, console.html, static/vendor, NOTICE; Paketdaten aus Aufgabe 1 bei Bedarf ergänzen. Assistentenantworten als Markdown mit Codeblöcken, Syntaxhervorhebung und KaTeX-Formeln rendern; Rohtext bleibt die einzige persistierte Inhaltsquelle. Generiertes HTML vor DOM-Einfügung sanitizen, aktives HTML und SVG nicht freigeben, gefährliche URL-Schemata blockieren und externe Bilder nicht automatisch laden. Codeinhalt niemals als HTML behandeln. Nutzernachrichten und Thinking standardmäßig als Text darstellen; Thinking separat aufklappbar machen. Streaming-Rendering bündeln, nicht bei jedem Token die gesamte Unterhaltung neu rendern; unvollständige Code- und Formelblöcke müssen lesbar bleiben. Copy-Aktionen für Nachricht, Markdown und einzelne Codeblöcke mit Clipboard-Fallback anbieten. Nur automatisch scrollen, solange der Nutzer am unteren Rand ist; sonst einen „Zum Ende“-Button anzeigen. Abnahme: Code und Formeln sind lesbar, Kopieren liefert Originalinhalt, bösartige Antworten/Importe lösen weder Script-Ausführung noch externe Bildrequests aus.

- [ ] **5. Chat-Themes und Bedienbarkeit ergänzen.** Dateien: static/chat.css, static/chat.mjs, console.html. Hell, Dunkel und System als gespeicherte Chat-Einstellung anbieten; Systemwechsel ohne Reload berücksichtigen. Die bestehenden Dashboardfarben nicht nebenbei neu gestalten. Tastaturbedienung, sichtbaren Fokus, Enter zum Senden und Shift+Enter für Zeilenumbruch erhalten; während IME-Komposition darf Enter nicht senden. Reduced-Motion berücksichtigen, Codeblöcke horizontal scrollbar machen und Lade-/Fehler-/Abbruchzustände klar anzeigen. Abnahme: schmale Ansichten und alle drei Themes bleiben lesbar; Funktionen sind ohne Maus erreichbar.

- [ ] **6. Streaming-Zustände vor Bearbeitung und Varianten absichern.** Dateien: static/chat.mjs, static/chat_state.mjs, tests/browser/chat_state.test.mjs. SSE-Dekodierung muss getrennte UTF-8-Zeichen, über Netzwerkchunks verteilte Events, Keepalive-Kommentare, Usage-only-Events und regulären Streamabschluss verarbeiten. Ein EOF ohne protokollkonformen Abschluss darf nicht als erfolgreiche Antwort gespeichert werden. Fehlgeschlagene Requests erhalten lokale Fehlerzustände statt Fehlermeldungen als Assistenteninhalt; Teilantworten bei Stop bleiben als abgebrochen erhalten. UI-Updates und Speicherung werden über Request-, Gesprächs- und Antwort-ID zugeordnet, sodass späte Events nach Stop oder Chatwechsel keine andere Antwort verändern. Uhr, Reader und Buttons auf jedem Exitpfad bereinigen. Abnahme: HTTP-Fehler, kaputte SSE-Payloads, Netzabbrüche, Reasoning-only-Ausgabe und expliziter Stop erzeugen konsistente Zustände und keinen dauerhaft aktiven Spinner.

- [ ] **7. Bearbeiten, Regenerieren und Antwortvarianten implementieren.** Dateien: static/chat_state.mjs, static/chat.mjs, static/chat.css, tests/browser/chat_state.test.mjs. Jede Nutzerfrage bildet einen Turn mit auswählbaren Assistentenvarianten. „Erneut generieren“ verwendet nur den ausgewählten gültigen Verlauf bis zur zugehörigen Nutzerfrage und erzeugt eine zusätzliche Variante, ohne ältere Antworten zu überschreiben. Variantenwechsel an einem frühen Turn mit späterem Verlauf erstellt nach Bestätigung eine Gesprächskopie bis zu diesem Turn; der ursprüngliche spätere Verlauf bleibt erhalten und wird nicht unter einem anderen Antwortpfad versendet. Bearbeiten einer früheren Nutzerfrage erstellt ebenfalls eine Gesprächskopie mit gültigem Prefix und korrigierter Frage, danach eine neue Antwort. So entsteht kein allgemeiner Gesprächsgraph. Auf Anfrage eine einmalige kreative Regeneration mit dokumentierter, auf 2 begrenzter Temperaturerhöhung anbieten; globale Standards unangetastet lassen. Abgebrochene Teilantworten nur nach expliziter Auswahl in einen späteren Request aufnehmen, reine Fehlerversuche nie. Abnahme: Requests enthalten ausschließlich den gewählten Pfad; alte Varianten und originale Gespräche bleiben erhalten; Edit/Regenerate verändern keine Serverdefaults.

### Phase 3 — Profile, Kontext und Austausch

- [ ] **8. Systemprompt-Profile und ehrlichen Modellstatus hinzufügen.** Dateien: static/chat_state.mjs, static/chat.mjs, static/chat.css, console.html, tests/browser/chat_state.test.mjs. Benannte Promptvorlagen erstellen, ändern, auswählen und löschen; Namen müssen nach Trimmen nicht leer und innerhalb der Profilliste eindeutig sein. Neue Chats können eine Standardvorlage übernehmen. Änderungen an Vorlagen wirken nicht rückwirkend auf gespeicherte Promptkopien. Das aktive Modell und dessen Fähigkeiten aus /v1/models anzeigen; die tatsächlich verwendete Modell-ID je Antwort speichern. Nach externem Modellwechsel oder Serverneustart Modellmetadaten aktualisieren und alte Kontextzählungen verwerfen. Bei einem Gespräch mit anderem bisherigen Modell vor erneutem Senden eine klare Bestätigung zeigen, dass das aktuell geladene Modell antworten wird. Keine vorgetäuschte Multi-Modell-Auswahl. Abnahme: Profile überleben Reload; frühere Gespräche behalten ihren Prompt; Modellabweichungen sind vor dem Senden sichtbar.

- [ ] **9. Exakte Kontextanzeige mit gemeinsamem Tokenizer integrieren.** Dateien: python/maxtoken/server/control_api.py, python/maxtoken/server/api_server.py, static/chat.mjs; neue tests/server/test_chat_count_tokens.py, ergänzend tests/browser/chat_state.test.mjs. Einen same-origin POST-Endpunkt /admin/chat/count-tokens hinzufügen, der OpenAI-förmige Chatrequests entgegennimmt. Konvertierung über chat_request_to_genspec und Anwendung der aktuellen Thinking-Vorgabe über apply_thinking_default wiederverwenden; anschließend count_prompt_tokens mit denselben Nachrichten und Templateparametern aufrufen. Keine zweite Tokenisierung oder Konvertierung über das Anthropic-Protokoll implementieren. Leere oder nicht-textuelle Prompts und Templatefehler als Clientfehler behandeln; Laden/Wartung als vorübergehende Nichtverfügbarkeit, Tokenizerfehler als Serverfehler. Antwort enthält input_tokens, context_length und tatsächlich geladenes model. Im Browser den gewählten Verlauf einschließlich Prompt und aktuellem Entwurf mit 500 ms Verzögerung zählen, alte Requests abbrechen und Ergebnisse mit überholter Gesprächs-/Modell-/Konfigurationsrevision verwerfen. Kontextanzeige umfasst Prompttokens, Prozent des aktuellen Fensters und das konfigurierte Ausgabebudget getrennt; sie behauptet keine exakte zukünftige Antwortlänge. Ab 85 Prozent warnen, bei überschrittenem verfügbaren Fenster eine verständliche Meldung zeigen. Keine automatische Kürzung von Gesprächen. Abnahme: gezählte Tokens entsprechen bei gleichem Prompt und gleichen Vorgaben den prompt_tokens der Generierung; fehlende oder veraltete Zählung erscheint als unbekannt, nicht als 0.

- [ ] **10. JSON-Import und -Export ohne Datenverlust ergänzen.** Dateien: static/chat_state.mjs, static/chat.mjs, console.html, tests/browser/chat_state.test.mjs. Einzelgespräche sowie den gesamten lokalen Bestand einschließlich Promptprofilen als versionierte JSON-Datei exportieren. Import nur für das eigene dokumentierte Format akzeptieren; direkte Kompatibilität mit oMLX-Dumps wird nicht versprochen. Maximale Importdateigröße auf 20 MiB begrenzen und vor dem Einlesen prüfen. Schema-Version, Texttypen, Rollen, IDs, Zeitangaben, Abschlussstatus und Variantenbezüge prüfen; aktive Streamzustände zu unterbrochen normalisieren. IDs neuer Kopien neu vergeben; Importe nach Vorschau und Bestätigung hinzufügen, nicht den bestehenden Bestand ersetzen. Gespeichertes HTML, eingebettete Bilder, fremde URLs als Attachments und unbekannte Formatversionen ablehnen. Syntaxfehler oder fehlgeschlagene Speicherungen dürfen bestehende Daten nicht ändern. Abnahme: Export und Reimport erhalten Inhalte, Varianten und Profile; fehlerhafte Dateien bleiben ohne Seiteneffekt; importierter Text durchläuft dieselbe sichere Darstellung.

### Phase 4 — Integration und Abnahme

- [ ] **11. Tests, Offline-Paketprüfung und Benutzerhinweise vervollständigen.** Dateien: tests/server/test_console.py, tests/server/test_chat_count_tokens.py, tests/browser/chat_state.test.mjs, README.md, pyproject.toml und NOTICE soweit nötig. Python-Tests für Assetauslieferung, Content-Types, Root-Redirect und Tokenzählung ergänzen; bestehende API-/Abbruch-/Thinking-Tests erhalten. Node-Tests für Formatvalidierung, Projektion des gewählten Gesprächspfads, Varianten, Edit-Kopien, Profile, Imports, Speicherfehler und streambedingte Zustandswechsel hinzufügen. Konkrete Browserabnahme dokumentieren: zwei Gespräche, Reload, Stream während Chatwechsel, Stop, Edit einer frühen Frage, Variantenwechsel mit Folgefragen, Copy von Code, bösartiges Markdown, Themewechsel, Modellabweichung, Context-Warnung, Export/Reimport und simulierter Quotenfehler. Frisch gebautes Wheel in isolierter Umgebung prüfen: alle Assets einschließlich KaTeX-Fonts sind enthalten; /console funktioniert vollständig ohne CDN. Hinweise zu browserlokaler Speicherung, Origin-/Portbindung, Export und nicht unterstützten Anhängen ergänzen. Abnahme: bestehende Serverfunktionen bleiben intakt und sämtliche untenstehenden Kriterien sind erfüllt.

### Abgegrenzte Folgearbeiten für volle oMLX-Parität

Diese Punkte sind keine versteckten Teilaufgaben der obigen Umsetzung und dürfen nicht als erledigt gemeldet werden:

| Feature | Fehlende Voraussetzung | Konsequenz für diesen Plan |
| --- | --- | --- |
| Modellwechsel aus dem Chat, Regeneration mit einem anderen Modell | Gesicherter Modellkatalog und browserzugängliche Lifecycle-Steuerung, die den Stop/Start des auf derselben Origin laufenden Servers überlebt; bestehende Daemon-Authentifizierung darf nicht umgangen werden. | Aktives Modell ehrlich anzeigen und externe Wechsel erkennen; tatsächliche Auswahl erst in einem eigenen Lifecycle-Integrationsplan. |
| Bild-/Videochat | Visionfähiger Loader, Prozessor und Inferenzpfad; momentan text-only Normalisierung. | Keine funktionslosen Uploadbuttons anzeigen. |
| Audioeingabe/-ausgabe | ASR-/TTS-Endpunkte und passende Modelle mit klaren Größen-/Laufzeitgrenzen. | Nicht implementieren oder durch bloße Textfelder vortäuschen. |
| PDF-, DOCX-, PPTX-Anhänge | Festgelegte lokale Textextraktion/OCR, Größenlimits und sichere Parser. | Eigenes Anhänge-Konzept notwendig; weder Binärdaten noch Dateipfade an den Textserver senden. |
| Websuche und autonome Toolrunden | Gesicherte Toolausführung, Berechtigungen und sichtbare Freigaben. | Nicht Bestandteil eines reinen Chat-UI-Umbaus. |
| Vollständige Oberfläche in mehreren Sprachen | Festgelegte Zielsprachen und Übersetzungsumfang. | Keine Einführung eines Übersetzungssystems nebenbei; bestehende englische UI-Sprache beibehalten. |

## Verification Criteria

- / und /console bleiben erreichbar; Console- und Logs-Ansichten behalten ihre bestehenden Funktionen.
- Die installierte Python-Distribution liefert alle Chatmodule, Styles, Vendor-Dateien und Fonts lokal aus.
- Chat, Codeformatierung und Formeln funktionieren ohne externe CDN-Requests.
- Gesprächsverlauf, Entwurf, ausgewählte Variante und Promptkopie bleiben nach Reload erhalten.
- Quotenfehler, beschädigte Daten, unbekannte Schema-Versionen und Tabkonflikte verursachen keinen stillen Datenverlust.
- Laufende Streams schreiben ausschließlich in das zugehörige Gespräch, auch wenn ein anderes sichtbar ist.
- Stop, HTTP-Fehler, Netzwerkfehler und unvollständige Streams führen zu unterscheidbaren, gespeicherten Zuständen.
- Regeneration erhält frühere Antworten; nur die ausgewählte Variante wird für Folgefragen übertragen.
- Bearbeiten und frühe Variantenwechsel erhalten das ursprüngliche Gespräch durch explizite Kopien.
- Promptprofile verändern bereits gespeicherte Gesprächsprompts nicht rückwirkend.
- Kreative Regeneration verändert ausschließlich den aktuellen Request, nicht die Serverdefaults.
- Generierte und importierte Inhalte können keine Scripts oder gefährlichen URLs ausführen und laden keine externen Bilder automatisch.
- Copy-Aktionen liefern den ursprünglichen Text oder Code, nicht sanitiziertes DOM-Markup.
- Die Kontextanzeige nutzt die tatsächliche Tokenisierung, das aktuelle Modellfenster und die aktuellen Thinking-Vorgaben.
- Modellabweichungen werden vor dem Senden angezeigt; eine Änderung des Requestnamens wird nicht als tatsächlicher Modellwechsel verkauft.
- Export und Reimport erhalten alle unterstützten Daten; invalides JSON und unbekannte Formate ändern den vorhandenen Bestand nicht.
- Tastaturbedienung, IME-Eingabe, mobile Breiten, Reduced-Motion und die drei Themes bleiben nutzbar.
- Nicht unterstützte Bild-, Audio-, Video- und Binärdatei-Funktionen werden nicht als verfügbar angezeigt.

## Risks and Mitigations

- **XSS durch Modellantworten und Importdateien:** Rohtext speichern, etablierte Sanitization, kein aktives HTML/SVG, URL-Schemata begrenzen und bösartige Beispiele in der Browserabnahme prüfen.
- **Verlust privater Gespräche:** Browserlokalität ausdrücklich erklären; Export anbieten; kein automatisches Kürzen wie bei oMLXs Quotenbehandlung; Schemafehler nicht durch Überschreiben kaschieren.
- **Konkurrierende Tabs und verspätete Streamevents:** stabile IDs, ein Request je Tab, revisionsgebundene Ergebnisse und Konfliktkopien verhindern stille Überschreibungen.
- **Falsche Modell- oder Kontextangaben:** serverseitige Metadaten verwenden, alte Zählungen invalidieren und Unterschiede offen anzeigen; kein Zeichen-pro-Token-Schätzwert als exakte Messung.
- **Langsame Browserdarstellung:** nur betroffene Antwort rendern, Updates bündeln, Suchindex zunächst ohne Zusatzabstraktion aus lokalem Text bilden und große Verläufe in der Browserabnahme prüfen.
- **Regressionsrisiko durch Auslagerung aus console.html:** kleine Chatmodule statt Gesamtumbau; vorhandene Dashboardtests und serverweite Einstellungssemantik erhalten.
- **Lizenz- und Lieferkettenrisiko:** Vendor-Versionen fixieren, jeweilige Lizenzen und erforderliche Hinweise übernehmen, oMLX-Branding nicht kopieren und keine externen Skripte zur Laufzeit laden.
- **Scheinsicherheit durch HTML-Stringtests:** reine Zustandslogik zusätzlich in Node testen; tatsächliches Rendering, Clipboard, Storage und Streaming im Browser abnehmen.

## Rejected Alternatives

- **Komplette oMLX-Chatdatei übernehmen:** eng an Alpine, Jinja, EnginePool, Auth, i18n und zahlreiche nicht vorhandene Endpunkte gebunden; unverhältnismäßige Integration und viele funktionslose Bedienelemente.
- **Swift zuerst:** löst die aktuellen Chat-Komfortlücken nicht und würde zwei Interfaces gleichzeitig etablieren.
- **React/Electron oder neues Buildsystem:** für die bestehende kleine Webkonsole unnötige Produktions- und Wartungsabhängigkeiten.
- **Markdown-/Sanitizer-/Formelparser selbst schreiben:** unnötiges Sicherheits- und Kompatibilitätsrisiko.
- **Serverseitige Chatsynchronisierung:** benötigt neue Datenschutz-, Authentifizierungs- und Speicherentscheidungen; browserlokale Speicherung genügt zunächst dem ausdrücklich untersuchten Komfortziel.
- **Allgemeiner Gesprächsbaum:** für Edit und Varianten unnötig komplex; explizite Gesprächskopien erhalten Daten mit einfacherer Semantik.
- **Alte Chats bei vollem Speicher automatisch löschen:** bequem für die Implementierung, aber unvertretbarer stiller Datenverlust.
- **Andere model-Namen als Modellwechsel behandeln:** passt nicht zum bestehenden Single-Modell-Server und würde Nutzern falsche Fähigkeiten suggerieren.
- **Anhänge allein im Frontend aktivieren:** Requestfelder ersetzen keine multimodale Inferenz oder sichere Dokumentextraktion.
