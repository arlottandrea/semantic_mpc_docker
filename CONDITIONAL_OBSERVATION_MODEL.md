# Modello di osservazione visuale condizionato per Semantic NMPC

## 1. Scopo

Questo documento definisce una possibile evoluzione del modello percettivo usato dal Semantic NMPC. L'obiettivo è sostituire i due surrogate MLP attuali, dipendenti soltanto dalla posa, con un unico modello condizionato anche da una rappresentazione dell'osservazione corrente.

L'idea centrale è:

1. estrarre da ogni albero visibile un latent visuale mediante un encoder condiviso;
2. mantenere tale latent costante durante un'ottimizzazione NMPC;
3. interrogare un piccolo MLP differenziabile in corrispondenza delle pose future previste;
4. stimare la probabilità e la qualità di una futura osservazione;
5. scegliere la traiettoria che massimizza la riduzione attesa dell'incertezza semantica, tenendo conto dei costi dinamici e geometrici.

Il modello non ricostruisce deterministicamente ciò che si trova nelle parti non osservate dell'albero. Impara invece una previsione statistica del tipo:

> Data l'osservazione corrente e la geometria rispetto all'albero, quanto è probabile ottenere una misura semantica utile da una determinata posa futura?

## 2. Limiti del modello corrente

Il modello attuale riceve, per ogni albero e posa prevista, un vettore equivalente a

```text
[dx_world, dy_world, yaw_assoluto]
```

Lo yaw viene successivamente rappresentato mediante seno e coseno. Questa formulazione presenta quattro limiti principali.

### 2.1 Nessun condizionamento sull'immagine corrente

Due scene visivamente differenti ma osservate dalla stessa posa producono la stessa predizione. Il surrogate può apprendere soltanto una mappa media dipendente dalla geometria.

### 2.2 Dipendenza dal riferimento globale

Uno yaw assoluto e coordinate relative espresse nel frame globale non costituiscono una descrizione geometricamente invariante. La qualità di un'osservazione dipende dall'orientamento della camera rispetto all'albero, non dall'orientamento assoluto nel mondo.

### 2.3 Due modelli separati

I modelli `raw` e `ripe` duplicano struttura e gestione. Il planner deve inoltre combinarne gli output per costruire il modello di misura.

### 2.4 Target fortemente degeneri

Dopo l'allineamento dei POV, ciascun dataset contiene 3240 record. La distribuzione di `tree_score` è:

| Dataset | `< 0.5` | `= 0.5` | `> 0.5` | Intervallo |
|---|---:|---:|---:|---:|
| raw | 148 | 3091 | 1 | `[0.17, 0.51]` |
| ripe | 9 | 3083 | 148 | `[0.33, 0.83]` |

Circa il 95% dei target vale esattamente `0.5`. Una regressione MSE non bilanciata ha quindi una soluzione quasi ottimale banale: predire sempre `0.5`.

Inoltre, il valore `0.5` attualmente può confondere fenomeni differenti:

- albero fuori dal campo visivo;
- frutti non rilevati;
- osservazione occlusa o troppo lontana;
- misura realmente ambigua fra raw e ripe.

Questi casi devono essere separati nel nuovo modello.

## 3. Variabili aleatorie e notazione

Per ogni albero `i` si definiscono:

- `C_i`: classe fisica persistente dell'albero;
- `b_i(c) = P(C_i = c)`: belief semantico corrente;
- `V_{i,t}`: disponibilità di un'osservazione utile al tempo `t`;
- `Y_{i,t}`: risultato della misura percettiva;
- `I_t`: immagine completa acquisita al tempo `t`;
- `I_{i,t}`: crop associato all'albero `i`;
- `z_{i,t}`: latent estratto dal crop;
- `m_{i,t}`: memoria visuale opzionale, aggregata nel tempo;
- `a_{i,t}`: posa anchor, cioè la posa dalla quale è stato prodotto il latent;
- `q_{i,k}`: posa query futura valutata dal NMPC.

La classe fisica resta binaria:

```text
C_i ∈ {raw, ripe}
```

La misura può invece avere tre esiti:

```text
Y_i ∈ {none, observed_raw, observed_ripe}
```

Questa distinzione è fondamentale. `None` non è normalmente una terza classe dell'albero: è un esito del processo di osservazione. Lo stesso albero ripe può produrre `observed_ripe` quando è ben visibile e `none` quando è dietro la camera.

Una terza classe semantica sarebbe corretta soltanto se rappresentasse uno stato fisico persistente, per esempio:

```text
C_i ∈ {raw, ripe, empty_tree}
```

Anche in quel caso `none` rimarrebbe un esito separato della misura.

## 4. Interpretazione del latent e della memoria

Il latent rappresenta lo stato visuale disponibile al planner al tempo corrente:

```text
z_{i,t} = Encoder(I_{i,t})
```

Durante una singola soluzione NMPC, `z_{i,t}` è un parametro costante. Le pose future sono invece variabili simboliche:

```text
prediction_{i,k} = MLP(z_{i,t}, a_{i,t}, q_{i,k})
```

Il latent costituisce la memoria dell'osservazione corrente. L'MLP non è una memoria in senso stretto: è un campo neurale condizionato, interrogato in pose differenti.

Se si vogliono integrare più osservazioni successive dello stesso albero, una singola immagine non è sufficiente. Si può introdurre una memoria ricorrente esterna a CasADi:

```text
e_{i,t} = Encoder(I_{i,t})
m_{i,t} = GRU(m_{i,t-1}, [e_{i,t}, a_{i,t}])
```

La GRU viene aggiornata soltanto quando arriva una nuova immagine. Durante l'orizzonte NMPC, `m_{i,t}` resta fisso e sostituisce `z_{i,t}`.

## 5. Rappresentazione geometrica

### 5.1 Coordinate relative all'oggetto

Siano la posa planare del robot

```text
x = [p_x, p_y, psi]
```

e la posizione dell'albero

```text
o_i = [o_x, o_y].
```

La differenza nel frame globale è

```text
d_world = o_i - p.
```

La differenza nel frame body/camera è

```text
d_body = R(-psi) d_world.
```

In alternativa si possono utilizzare distanza e bearing relativo:

```text
r = ||d_world||
alpha = wrap(atan2(d_y, d_x) - psi - camera_yaw_offset)
features = [r, sin(alpha), cos(alpha)].
```

Queste rappresentazioni eliminano la dipendenza artificiale dallo yaw assoluto.

### 5.2 Anchor e query

Il modello deve conoscere sia la geometria associata al latent sia la geometria futura:

```text
anchor_features = geometry(robot_t, tree_i)
query_features  = geometry(robot_k, tree_i)
```

Una possibile interfaccia è:

```text
MLP([memory_i,
     anchor_dx_body, anchor_dy_body,
     query_dx_body,  query_dy_body])
```

Se l'orientamento della camera non è determinato implicitamente dalla posizione, occorre aggiungere seno e coseno del bearing relativo. Se invece una camera orientabile viene sempre puntata verso l'albero, l'orientamento può essere rimosso dal modello di misura.

### 5.3 Una rotazione di 180 gradi

Una query ruotata di 180 gradi rispetto all'anchor non è automaticamente non visibile. La visibilità dipende dalla camera rispetto all'albero nella posa query.

Per esempio, il drone può spostarsi sul lato opposto dell'albero e ruotare di 180 gradi continuando a guardarlo. Il criterio corretto usa `alpha_query`, non soltanto `delta_yaw = yaw_query - yaw_anchor`.

## 6. Associazione fra alberi e crop visuali

Le posizioni note degli alberi consentono un'associazione object-centric. Per ogni albero si proietta una bounding box 3D nel piano immagine.

Dato un punto espresso nel frame camera,

```text
P_camera = [X, Y, Z],
```

la proiezione pinhole è

```text
u = f_x X / Z + c_x
v = f_y Y / Z + c_y.
```

Se `Z <= 0`, il punto si trova dietro il piano della camera. Per stimare un crop sono necessari:

- trasformazione TF da `map` a camera;
- matrice intrinseca della camera;
- posizione 3D dell'albero;
- altezza e raggio approssimativi, oppure una bounding box 3D nota;
- intersezione della bounding box proiettata con i limiti dell'immagine.

Il nodo di data association corrente esegue principalmente il percorso inverso:

```text
detection 2D -> profondità -> punto 3D -> frame map -> albero più vicino.
```

La nuova pipeline aggiunge:

```text
albero noto -> frame camera -> proiezione 2D -> crop indicizzato per albero.
```

I due percorsi possono essere usati insieme: la proiezione genera la ROI prevista, mentre detection e profondità verificano contenuto e occlusione.

### 6.1 Immagini nere e latent nullo

Non è consigliato rappresentare un albero fuori FOV mediante un'immagine completamente nera. La rete imparerebbe una correlazione artificiale che non corrisponde alle immagini reali, nelle quali sono presenti cielo, terreno, vegetazione o altri oggetti.

La rappresentazione raccomandata è:

```text
tree visible     -> z_i = Encoder(crop_i), has_observation_i = 1
tree not visible -> z_i = z_null,          has_observation_i = 0
```

`z_null` può essere appreso o inizializzato a zero. Il flag esplicito evita che un latent numericamente vicino allo zero venga confuso con l'assenza di osservazione.

Per addestrare la robustezza percettiva sono invece utili negativi reali:

- ROI contenenti sfondo;
- alberi parzialmente fuori immagine;
- alberi occlusi;
- alberi nel FOV ma senza detection;
- immagini a distanza elevata o degradate.

## 7. Architettura raccomandata

La soluzione è divisa in una parte esterna e una interna a CasADi.

```text
                         fuori da CasADi
RGB + TF + tree poses
        |
        v
projection -> crop per-tree -> shared encoder -> memory_i
                                                   |
                                                   | parametro fisso
                                                   v
query pose X_k -> object-relative geometry -> Conditional MLP
                                                   |
                                                   v
                                      visibility + semantic quality
                                                   |
                                                   v
                                          likelihood 2 x 3
                                                   |
                                                   v
                                 expected semantic information gain
                                                   |
                                                   v
                                           costo del NMPC
```

### 7.1 Fuori da CasADi

```text
RGB image
    -> projection/crops per tree
    -> shared visual encoder
    -> latent projection, per esempio 16-64 dimensioni
    -> optional recurrent memory
    -> one fixed memory vector per tree
```

Con il dataset disponibile è preferibile partire da un encoder visuale pretrained anziché addestrare un autoencoder da zero. Possibili baseline sono DINOv2 ViT-S/14 o una variante piccola di ConvNeXt V2. L'encoder può essere inizialmente congelato e successivamente sottoposto a fine-tuning parziale.

Un decoder non è necessario per l'inferenza. Una loss di ricostruzione può essere aggiunta come regolarizzazione ausiliaria, ma non garantisce che il latent conservi le informazioni rilevanti per la pianificazione.

### 7.2 Dentro CasADi

CasADi contiene soltanto un piccolo head differenziabile:

```text
ConditionalMLP(memory_i, anchor_features_i, query_features_i)
    -> visibility_i
    -> semantic_quality_i
```

Una configurazione iniziale ragionevole è:

```text
latent dimension: 32
hidden layers: 2
hidden width: 64
output activations: sigmoid
```

Il latent non è una variabile di decisione. Entra nei parametri dell'ottimizzazione e viene ripetuto per le query dell'orizzonte. Le derivate necessarie al solver riguardano quindi la posa query attraverso il piccolo MLP.

## 8. Parametrizzazione del modello di misura

### 8.1 Separazione fra visibilità e semantica

La rete produce:

```text
v_i(q) = P(observation available | memory_i, anchor_i, q)
```

e una qualità semantica condizionata alla visibilità. Per un sensore simmetrico si può usare un solo parametro:

```text
kappa_i(q) = P(correct semantic label | visible, memory_i, anchor_i, q).
```

La likelihood `2 x 3` diventa:

```text
                         Y = none    Y = raw       Y = ripe
C = raw                    1-v        v*kappa       v*(1-kappa)
C = ripe                   1-v        v*(1-kappa)   v*kappa
```

Se il detector ha prestazioni asimmetriche, l'MLP può produrre due accuratezze:

```text
kappa_raw  = P(observed_raw  | C=raw,  visible)
kappa_ripe = P(observed_ripe | C=ripe, visible)
```

ottenendo:

```text
L_raw  = [1-v, v*kappa_raw,      v*(1-kappa_raw)]
L_ripe = [1-v, v*(1-kappa_ripe), v*kappa_ripe]
```

Questa struttura usa un solo modello condiviso e garantisce che la no-detection abbia la stessa probabilità sotto entrambe le ipotesi semantiche.

### 8.2 Perché `none` non deve essere la terza classe fisica

Un classificatore percettivo può legittimamente avere un softmax a tre output:

```text
[P(none), P(observed_raw), P(observed_ripe)].
```

Tale vettore descrive l'esito della prossima osservazione, non lo stato persistente dell'albero. Inserire `none` direttamente nel belief dell'oggetto confonderebbe una proprietà del sensore con una proprietà del mondo.

## 9. Aggiornamento bayesiano ed entropia

Sia il prior binario

```text
b(c) = P(C=c), c ∈ {raw, ripe}
```

e sia `L(c,y) = P(Y=y | C=c)` la likelihood `2 x 3`.

La probabilità predittiva dell'osservazione è:

```text
P(Y=y) = sum_c b(c) L(c,y).
```

Il posterior dopo aver osservato `y` è:

```text
b(c | y) = b(c) L(c,y) / P(Y=y).
```

L'entropia del belief semantico è:

```text
H(b) = -sum_c b(c) log2(b(c)).
```

L'entropia attesa dopo una futura misura è:

```text
E[H_after] = sum_y P(Y=y) H(b(. | y)).
```

L'information gain è:

```text
IG = H(b) - E[H_after].
```

L'entropia rimane quindi binaria, anche se la misura possiede tre esiti. Non si deve minimizzare l'entropia del vettore `[none, observed_raw, observed_ripe]`: quella quantità misura l'incertezza sul risultato del sensore, non l'incertezza sulla classe dell'albero.

### 9.1 Dimostrazione per la no-detection

Con la fattorizzazione proposta:

```text
L(raw, none) = L(ripe, none) = 1-v.
```

Il posterior dopo `none` è:

```text
b(c | none)
  = b(c)(1-v) / sum_c' b(c')(1-v)
  = b(c).
```

Una no-detection non modifica quindi il belief e non riduce l'entropia. Se `v=0`, l'unico esito possibile è `none` e l'information gain è esattamente zero. Il planner non riceve alcun incentivo informativo ad andare verso una zona non osservabile.

Se si consentisse

```text
P(none | raw) != P(none | ripe),
```

l'assenza diventerebbe evidenza semantica. Questo può essere statisticamente reale quando il detector ha sensibilità diverse per le due classi, ma può anche produrre scorciatoie e comportamenti indesiderati. La parametrizzazione a visibilità condivisa lo esclude deliberatamente.

## 10. Funzione costo NMPC

Per un orizzonte `N`, una funzione costo indicativa è:

```text
J = sum_k (
        w_motion * motion_cost_k
      + w_yaw * yaw_cost_k
      + w_acc * acceleration_cost_k
      + w_distance * target_distance_cost_k
      - w_info * gamma^k * IG_k
    )
```

soggetta a dinamica, limiti di velocità e accelerazione, limiti del campo e vincoli di collisione.

L'entropia non è quindi l'unico termine. È il termine che quantifica il valore informativo della traiettoria. Gli altri termini determinano fattibilità, regolarità e costo del movimento.

Una forma semplificata, utile come prima baseline, è:

```text
IG_k ~= v_k * IG_semantic_when_visible_k.
```

Questa espressione rende esplicito che la visibilità modula la possibilità di ottenere informazione ma non è essa stessa informazione semantica.

### 10.1 Osservazioni ripetute nell'orizzonte

Sommare `IG_k` calcolati tutti rispetto allo stesso prior può contare più volte la stessa informazione. La soluzione esatta richiederebbe propagare un albero di belief su tutti i possibili esiti, con costo esponenziale.

Sono possibili tre approssimazioni:

1. limitare l'information gain cumulativo all'entropia corrente dell'albero;
2. applicare un forte discount temporale;
3. propagare un belief atteso deterministico, accettando l'approssimazione.

Il limite superiore

```text
sum_k IG_k <= H(b_t)
```

è necessario perché nessuna sequenza di osservazioni può rimuovere più incertezza di quella inizialmente presente.

## 11. Evitare il doppio conteggio dell'immagine corrente

L'immagine al tempo `t` svolge due ruoli differenti:

1. aggiorna immediatamente il belief semantico `b_{i,t}`;
2. genera il latent usato per prevedere la qualità delle osservazioni future.

Non deve essere trattata nuovamente come evidenza indipendente dentro l'orizzonte. In particolare, un latent che codifica direttamente la classe ripe/raw potrebbe consentire all'MLP di ripetere la stessa informazione a ogni query.

La formulazione raccomandata limita il ruolo del latent alla predizione di:

- visibilità futura;
- occlusione e qualità del viewpoint;
- affidabilità attesa del classificatore.

Il belief semantico resta una variabile esplicita aggiornata una sola volta per ogni nuova misura.

Possibili regolarizzazioni sono:

- bottleneck latent piccolo;
- stop-gradient fra semantic classifier e observation-quality head;
- loss avversaria per ridurre l'informazione di classe nel latent di qualità;
- output strutturato `v`, `kappa_raw`, `kappa_ripe` invece di un posterior semantico diretto.

## 12. Costruzione del dataset anchor-query

Il training deve essere multi-view. Usare la stessa immagine come input e target permetterebbe alla rete di ignorare la posa e leggere direttamente la confidenza corrente.

Ogni esempio deve contenere:

```text
tree_id
scene_id
class_label
anchor_image
anchor_pose_relative_to_tree
query_pose_relative_to_tree
query_visibility
query_detector_result
```

Il forward pass è:

```text
z_anchor = Encoder(anchor_crop)
prediction = MLP(z_anchor, anchor_pose, query_pose)
loss = measurement_loss(prediction, target_at_query)
```

Per ogni anchor si campionano query:

- vicine e lontane;
- con piccoli e grandi cambiamenti di bearing;
- completamente visibili;
- parzialmente visibili;
- occluse;
- fuori FOV;
- dietro la camera.

Gli anchor usati dall'encoder dovrebbero contenere almeno una parte reale dell'albero. Le query non visibili devono comunque essere presenti per supervisionare `v=0`; non richiedono un'immagine nera.

### 12.1 Uso dei dataset raw e ripe allineati

I due dataset ora contengono gli stessi 3240 POV nello stesso ordine. Questo consente:

- campionamento simmetrico delle pose;
- confronto raw/ripe a parità di viewpoint;
- bilanciamento delle classi;
- stima della matrice di confusione condizionata alla posa.

Occorre però verificare l'identità della scena e dell'albero prima di creare coppie anchor-query. Non si devono accoppiare immagini di alberi o scene differenti soltanto perché hanno coordinate numericamente simili.

### 12.2 Target e loss

È preferibile conservare separatamente:

- ground truth della classe fisica;
- visibilità geometrica;
- presenza di una detection;
- score raw e ripe del detector;
- eventuale occlusione.

Una loss strutturata può essere:

```text
L = lambda_v * BCE(v_pred, v_target)
  + lambda_s * v_target * semantic_NLL_or_KL
  + lambda_cal * BrierScore
```

`semantic_NLL_or_KL` viene applicata soltanto quando esiste una misura semanticamente valida. Il Brier score o una loss di calibrazione sono utili perché l'NMPC utilizza direttamente le probabilità, non soltanto la classe argmax.

## 13. Split e validazione

Uno split casuale per singola immagine produrrebbe leakage: POV adiacenti sono fortemente correlati. Train, validation e test devono essere separati per almeno uno dei seguenti identificatori:

- albero;
- scena;
- traiettoria;
- intervallo spaziale;
- configurazione di illuminazione o simulazione.

Metriche percettive:

- visibility AUROC/AUPRC;
- negative log-likelihood;
- Brier score;
- expected calibration error;
- accuracy raw/ripe condizionata alla visibilità;
- errore rispetto a distanza e bearing.

Metriche di pianificazione:

- riduzione reale dell'entropia per metro percorso;
- tempo necessario a raggiungere una confidenza target;
- percentuale di alberi classificati correttamente;
- lunghezza e regolarità della traiettoria;
- tempo di soluzione IPOPT;
- robustezza a latent nullo e detection mancanti.

Ablation minima:

1. modello pose-only attuale;
2. latent singolo + posa;
3. memoria ricorrente + posa;
4. visibilità analitica contro visibilità appresa;
5. head simmetrico contro matrice di confusione asimmetrica.

## 14. Integrazione runtime proposta

Per ogni nuovo frame:

1. acquisire RGB, depth, camera info e trasformazioni TF coerenti temporalmente;
2. ottenere le pose degli alberi;
3. proiettare le bounding box previste nel frame camera;
4. estrarre e ridimensionare i crop validi;
5. calcolare un latent per ogni albero visibile;
6. aggiornare la memoria per-tree e il flag `has_observation`;
7. aggiornare il belief mediante la misura corrente;
8. passare a CasADi belief, memoria, anchor e maschere come parametri;
9. valutare il Conditional MLP sulle pose query dell'orizzonte;
10. ottimizzare expected information gain e costi dinamici;
11. eseguire il primo controllo e ripetere al frame successivo.

Per un albero mai osservato:

```text
memory_i = z_null
has_observation_i = 0
```

Il planner usa un modello medio pose-only o un prior appreso. Appena l'albero entra nel FOV, il latent nullo viene sostituito dalla rappresentazione reale.

## 15. Integrazione CasADi proposta

Il vettore dei parametri dell'ottimizzatore deve includere, per ogni target selezionato:

```text
tree_position:       2 valori
semantic_belief:     2 valori
target_mask:         1 valore
memory:              latent_dim valori
anchor_features:     anchor_dim valori
has_observation:     1 valore
```

Il batch simbolico del modello diventa concettualmente:

```text
for horizon step k:
    for target tree i:
        query = geometry(X_k, tree_i)
        model_input = [memory_i, anchor_i, query, has_observation_i]
```

L'output viene convertito nella likelihood strutturata `2 x 3`, quindi utilizzato per l'entropia attesa.

Per evitare rallentamenti del solver:

- mantenere il latent piccolo;
- non inserire l'encoder in CasADi;
- usare un head con poche operazioni e attivazioni supportate;
- batchare target e orizzonte in una sola chiamata L4CasADi;
- misurare Jacobian/Hessian e tempo IPOPT rispetto alla baseline;
- valutare una parametrizzazione low-rank o FiLM se la concatenazione del latent è troppo costosa.

## 16. Visibilità analitica e appresa

La visibilità può essere separata in due fattori:

```text
v(q) = v_geometric(q) * v_learned(memory, anchor, q).
```

`v_geometric` elimina casi impossibili, come albero dietro la camera, fuori range o fuori FOV. Per mantenere differenziabilità si possono usare gate sigmoidali morbidi basati su distanza e bearing.

`v_learned` modella fenomeni non catturati dalla geometria semplice:

- occlusione da chioma o rami;
- densità locale dei frutti;
- qualità dell'immagine;
- viewpoint più o meno informativi;
- errori sistematici del detector.

Questa fattorizzazione riduce il carico di apprendimento e impedisce alla rete di assegnare information gain a pose geometricamente impossibili.

## 17. Decisioni raccomandate

La prima implementazione dovrebbe adottare le seguenti scelte:

1. belief fisico binario `{raw, ripe}`;
2. esiti della misura `{none, observed_raw, observed_ripe}`;
3. encoder condiviso fuori da CasADi;
4. latent object-centric per ogni albero;
5. latent nullo e mask, non immagini nere;
6. anchor e query espresse rispetto all'albero;
7. visibilità geometrica moltiplicata per qualità appresa;
8. un solo Conditional MLP dentro CasADi;
9. likelihood strutturata con `P(none | raw) = P(none | ripe)`;
10. expected posterior entropy sul belief binario;
11. training cross-view anchor-query;
12. aggiornamento del belief corrente separato dalla previsione futura;
13. memoria ricorrente soltanto dopo aver validato la baseline con latent singolo.

## 18. Piano incrementale

### Fase A: dataset e geometria

- aggiungere `tree_id`, `scene_id` e label di visibilità;
- implementare la proiezione tree-to-image;
- generare crop e coppie anchor-query;
- distinguere `none` da ambiguità semantica;
- realizzare split senza leakage.

### Fase B: modello offline

- baseline pose-only con il nuovo target;
- encoder frozen + conditional head;
- calibrazione delle probabilità;
- confronto latent contro pose-only;
- test su query con grandi cambiamenti di viewpoint.

### Fase C: runtime ROS

- estrazione crop per-tree;
- cache dei latent;
- aggiornamento del belief corrente;
- pubblicazione di latent, mask e diagnostica della visibilità.

### Fase D: NMPC

- estendere i parametri CasADi;
- sostituire i due modelli con un head condiviso;
- estendere l'entropia attesa da due a tre esiti della misura;
- verificare che `v=0` produca `IG=0` numericamente;
- confrontare tempo di soluzione e traiettorie.

### Fase E: memoria temporale

- introdurre GRU o aggregazione attention per-tree;
- verificare che osservazioni successive migliorino calibrazione e planning;
- controllare che la stessa evidenza non venga contata più volte.

## 19. Conclusione

L'idea di mantenere fisso un latent dell'osservazione corrente e variare la posa query dentro il NMPC è coerente. Il risultato è un campo di osservazione implicito e condizionato, capace di stimare dove una futura misura sarà più utile.

La validità del sistema dipende però da quattro condizioni:

1. training cross-view, non auto-predizione della stessa immagine;
2. rappresentazione object-centric con associazione per-tree;
3. separazione fra classe fisica, visibilità ed esito della misura;
4. calcolo dell'information gain sul belief semantico, non sull'entropia dei tre output percettivi.

Con queste condizioni, il modello può usare l'informazione visuale corrente senza inserire l'encoder nel solver e senza conservare due MLP distinti per raw e ripe.
