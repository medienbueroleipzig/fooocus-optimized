##Warum läuft es nicht auf MPS und was kann man tun?
Dein M3-Chip unterstützt keine CUDA, da CUDA ausschließlich auf NVIDIA-GPUs läuft. Stattdessen nutzt Apple sein eigenes MPS-Backend, das zwar viele, aber noch nicht alle Funktionen von PyTorch vollständig unterstützt.

##Mögliche Lösungen:
Alternative Implementierungen: Falls dein Code stark von std_mean.correction abhängt, kannst du möglicherweise eine alternative Berechnung oder eine Bibliothek verwenden, die besser auf MPS läuft.
Warten auf zukünftige PyTorch-Updates: PyTorch arbeitet laufend daran, die Unterstützung für MPS zu verbessern. Zukünftige Updates könnten diese Funktion auf MPS unterstützen.
CPU-Optimierung: Wenn es nicht möglich ist, auf die GPU zuzugreifen, könnte eine Optimierung auf der CPU-Seite helfen. Ein leistungsfähiger Chip wie der M3 mit 36 GB Speicher sollte recht effizient arbeiten können, auch wenn es nicht so schnell ist wie eine voll unterstützte GPU.
4. Die Progarmamierlösung bedeutet, dass man im Quellcode der Python anwendung alle torch.std  und damit std_main Vorkommen mit einer "manuellen". Berechnung des Standardwertes (das macht std) ersetzt. Das heisst, man muss diese Operation durch separate Berechnungen für torch.std und torch.mean ersetzen. Das könnte verhindern, dass diese Operation auf die CPU ausgelagert wird und damit potenziell die Inferenzgeschwindigkeit verbessern.
Zusammengefasst:

Um das Problem mit aten::std_mean.correction zu lösen, kannst du versuchen, die torch.std_mean-Operation in getrennte Aufrufe von torch.std und torch.mean umzuwandeln. Dadurch wird der CPU-Fallback vermieden und die Operation kann möglicherweise vollständig auf der MPS-GPU ausgeführt werden. Hier ist, wie du das in anisotropic.py anpassen kannst:

Schritt-für-Schritt-Anleitung zur Anpassung
Öffne die Datei anisotropic.py:
Finde die Zeile, in der torch.std_mean verwendet wird. In deiner Ausgabe steht die Zeile in etwa so:

's, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)'
Ersetze 'torch.std_mean' durch separate torch.std und torch.mean Berechnungen:
Ändere die Zeile in zwei separate Aufrufe für torch.std und torch.mean, sodass sie wie folgt aussieht:
's = torch.std(g, dim=(1, 2, 3), keepdim=True)
m = torch.mean(g, dim=(1, 2, 3), keepdim=True)'
Diese Änderung berechnet den Standardabweichungs- und den Mittelwert getrennt und vermeidet somit den CPU-Fallback, da torch.std und torch.mean auf MPS unterstützt werden.
Speichern und testen:
Speichere die Änderungen in anisotropic.py und starte Fooocus erneut.
Achte darauf, ob die Warnung für aten::std_mean.correction verschwindet und beobachte, ob sich die Inferenzgeschwindigkeit verbessert.
# Berechnung für ro_pos
'mean_pos = torch.mean(cond, dim=(1, 2, 3), keepdim=True)
var_pos = torch.mean((cond - mean_pos) ** 2, dim=(1, 2, 3), keepdim=True)
ro_pos = torch.sqrt(var_pos)'
# Berechnung für ro_cfg
'mean_cfg = torch.mean(x_cfg, dim=(1, 2, 3), keepdim=True)
var_cfg = torch.mean((x_cfg - mean_cfg) ** 2, dim=(1, 2, 3), keepdim=True)
ro_cfg = torch.sqrt(var_cfg)'
Hier sind spezifische Anpassungen, die du an den Stellen mit torch.std und torch.std_mean vornehmen kannst, um die CPU-Fallbacks zu vermeiden und die MPS-Leistung zu optimieren.

*1. Anpassung in anisotropic.py
In der Funktion adaptive_anisotropic_filter wird torch.std_mean verwendet:

's, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)'
Lösung
Ersetze torch.std_mean durch separate Berechnungen für torch.mean und torch.var, um die Standardabweichung zu berechnen:

'm = torch.mean(g, dim=(1, 2, 3), keepdim=True)
var = torch.mean((g - m) ** 2, dim=(1, 2, 3), keepdim=True)
s = torch.sqrt(var + 1e-5)  # Kleine Konstante hinzufügen, um Division durch 0 zu vermeiden'
Diese Änderung sollte dazu führen, dass die Operation vollständig auf der MPS-GPU bleibt.

2. Anpassung in external_model_advanced.py
Hier finden wir zwei torch.std-Verwendungen in der Klasse RescaleCFG:

'ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)'
Lösung
Ersetze auch hier torch.std durch die Kombination aus Mittelwert und Varianz:

'# Berechnung für ro_pos
mean_pos = torch.mean(cond, dim=(1, 2, 3), keepdim=True)
var_pos = torch.mean((cond - mean_pos) ** 2, dim=(1, 2, 3), keepdim=True)
ro_pos = torch.sqrt(var_pos)
# Berechnung für ro_cfg
mean_cfg = torch.mean(x_cfg, dim=(1, 2, 3), keepdim=True)
var_cfg = torch.mean((x_cfg - mean_cfg) ** 2, dim=(1, 2, 3), keepdim=True)
ro_cfg = torch.sqrt(var_cfg)'
3. Anpassung in supported_models.py
In der Datei supported_models.py wird torch.std folgendermaßen verwendet:

'if torch.std(out, unbiased=False) > 0.09:  # not sure how well this will actually work. I guess we will find out.'
Lösung
Auch hier kann die Standardabweichung durch separate Mittelwert- und Varianzberechnungen ersetzt werden:

'mean_out = torch.mean(out)
var_out = torch.mean((out - mean_out) ** 2)
std_out = torch.sqrt(var_out)
if std_out > 0.09:'

*Test und Beobachtungen
Diese Anpassungen sollten den CPU-Fallback vermeiden und die Inferenzleistung auf MPS verbessern. Nach diesen Änderungen empfehle ich, die Anwendung zu starten und die Logs zu beobachten, um sicherzustellen, dass die CPU-Fallback-Warnungen verschwunden sind und keine neuen Fehler auftauchen.

Durch diese Aufteilung in einzelne Berechnungen für Mittelwert und Standardabweichung kann die Berechnung vollständig auf der GPU bleiben, was zu einer spürbaren Verbesserung der Performance führen sollte.

*Warum diese Anpassung funktioniert
Der kombinierte Aufruf torch.std_mean ist auf der MPS-Architektur (Apple Silicon) derzeit nicht vollständig unterstützt und führt daher zum CPU-Fallback. Indem du die Operationen auf torch.std und torch.mean aufteilst, stellst du sicher, dass die Berechnungen vollständig auf der MPS-GPU ausgeführt werden, da diese beiden Operationen separat auf MPS unterstützt werden.

*Mögliche Auswirkungen
Diese Anpassung sollte den CPU-Fallback vermeiden und die Berechnungen auf der MPS-GPU halten, was zu einer höheren Inferenzgeschwindigkeit führen kann.
Die Genauigkeit und das Ergebnis der Berechnungen sollten unverändert bleiben, da die Operationen mathematisch äquivalent sind.
Wenn du diese Änderung durchgeführt hast, kannst du testen, ob Fooocus schneller arbeitet und die CPU-Fallback-Warnung verschwunden ist.

 
*Der Fehler tritt auf, weil aten::std_mean.correction von MPS nicht unterstützt wird und daher auf die CPU zurückfällt.
 
CUDA und NVIDIA-GPUs bieten eine bessere Unterstützung für eine breite Palette von Operationen, die auf Apple Silicon noch eingeschränkt sind.
 
Mögliche Lösungen umfassen alternative Berechnungsstrategien oder Abwarten auf zukünftige PyTorch-Updates, die mehr Unterstützung für MPS bieten könnten.
