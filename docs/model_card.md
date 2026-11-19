# Model cards

The game uses two models: a gesture reader that turns hand landmarks into a move, and a move predictor that learns each player during a game. Every number here comes from [evaluation_report.md](evaluation_report.md).

## Gesture reader

| Item | Value |
| --- | --- |
| Type | Multilayer perceptron on MediaPipe hand landmarks |
| Input | The 21 landmarks of one hand, rotated, scaled, and mirrored into a standard position: 40 numbers |
| Output | Probabilities of rock, paper, scissors, and not a move |
| Architecture | Two hidden layers of 64 ReLU units, softmax output, 7,044 parameters |
| Training | scikit-learn `MLPClassifier` with Adam, L2 penalty 0.0001, and early stopping on a held-out tenth. 80,000 hands from 17,473 people, each used twice with a random vertical stretch between 0.5 and 2. |
| Decision | The most likely class, when its probability is at least 0.81; otherwise the game asks again. In play, the frames after "Shoot!" vote. |
| File | `web/models/gesture_reader.json`, 70 KB, weights rounded to six significant digits |
| Licence | CC BY-SA 4.0, as an adaptation of the HaGRID annotations |

### Intended use

- **Intended:** reading one deliberate rock, paper, or scissors gesture in the game.
- **Out of scope:** sign language, general gesture control, and any decision about a person. The reader was trained on 22 HaGRID gesture labels grouped into four classes.

### Data

Hand landmarks from the HaGRID v2 annotations, which MediaPipe produced from photos of people making gestures. The project uses HaGRID's split by person:

| Split | Hands | People |
| --- | --- | --- |
| Training | 80,000 | 17,473 |
| Validation | 11,923 | 1,300 |
| Test | 19,877 | 2,236 |

Each class has the same number of hands, spread evenly over its HaGRID labels. The labels in each class are listed in [architecture.md](architecture.md#the-training-data).

### Evaluation

On the 19,877 test hands, from people the model never saw:

| Metric | Neural network | Finger-counting rule |
| --- | --- | --- |
| Accuracy | 99.43% | 89.75% |
| Moves read as the right move | 99.46% | 99.44% |
| Moves the game would ask to see again | 0.50% | 0.40% |
| Non-moves played as a move | 0.66% | 39.08% |
| Hands played as a move the player did not show | 0.20% | 9.96% |

The finger-counting rule reads moves as well as the network, but it cannot tell a thumbs up or a relaxed hand from a move. Logistic regression reaches 98.01% accuracy but has to reject 39% of real moves to make as few wrong moves.

The weakest classes are relaxed hands with no gesture, of which 13.8% fall below the threshold and 2.8% are played as a move, and pointing, which is read as rock 2.6% of the time. Of the gestures left out of training, three fingers (thumb, index, middle) is read as scissors 88% of the time and a loose grabbing hand as paper 55% of the time.

Accuracy stays at 99.4% for every vertical stretch from 0.5 to 2, so the unknown aspect ratio of the training photos does not matter.

### Limitations

- **Photos, not webcam video.** The test hands come from still photos. Webcam frames add motion blur, low light, and hands half out of frame, which the evaluation does not cover. Voting over several frames softens this but has not been measured.
- **A different MediaPipe.** HaGRID's landmarks were made with an earlier MediaPipe release than the one in the game. The landmark layout is the same, but small differences in placement are possible.
- **Not evaluated across groups.** Accuracy has not been broken down by skin tone, age, or hand size. HaGRID's estimated age, gender, and race labels come from other networks and were not used. Hands that MediaPipe fails to detect never reach the reader, so detection gaps would show up as "No hand in view" rather than as wrong moves.
- **One hand at a time.** The landmarker follows one hand. With two hands in view, it may pick either.

## Move predictor

| Item | Value |
| --- | --- |
| Type | Seven online experts combined with exponential weights on their log loss |
| Input | The moves of both sides in the current game so far |
| Output | Probabilities of the player's next move. The AI plays the move with the best expected score against them. |
| Training | None before a game: each game starts from scratch. Five settings were tuned on 250 recorded players: count decay 0.9, prior count 1, learning rate 1, score decay 0.95, margin 0. |
| File | `web/models/predictor.json` |

The experts and the combiner are described in [architecture.md](architecture.md#predicting-the-next-move).

### Intended use

- **Intended:** playing rock, paper, scissors against people, and showing them which of their habits it picked up.
- **Out of scope:** any claim about a person beyond their play in one game.

### Evaluation

On 249 recorded players it never saw, each playing 50 rounds, with 95% bootstrap intervals over players:

| AI strategy | Won | Drawn | Lost | Net score |
| --- | --- | --- | --- | --- |
| Random | 33.6% (32.8 to 34.5) | 33.0% | 33.4% | +0.3% |
| Beat the last move | 27.2% (25.4 to 28.9) | 35.9% | 37.0% | -9.8% |
| After the last two moves, alone | 42.4% (40.9 to 44.0) | 30.3% | 27.3% | +15.0% |
| All predictors combined | 42.3% (40.8 to 43.9) | 30.6% | 27.1% | +15.2% |

The combined AI wins 36.7% of rounds 1 to 10 and 46.1% of rounds 41 to 50. It matches the best single expert on wins and gives better-calibrated probabilities, with a log loss of 1.0504 against 1.0642. Its advantage is that it does not need to know in advance which habit a player has.

### Limitations

- **Replayed, not played.** The recorded players faced a different computer program. The evaluation shows how well the AI predicts people's moves, not how they would respond to an opponent that adapts to them. A player who notices the AI's habits can change theirs.
- **It can be beaten.** A player who chooses truly at random holds it to a third of rounds. When the AI is confident it plays the counter to its forecast without randomising, so a player who works out the forecast can exploit it.
- **One dataset.** All recorded games come from one study, with 50 rounds per player. The data does not document how players were recruited or instructed, or which number stands for which move. The [evaluation report](evaluation_report.md#how-the-recorded-players-played) shows that the encoding barely changes the results.
- **Simplified explanations.** The sentence under "Why" describes the expert with the largest weight. The forecast is a weighted average of all seven, so other experts also contribute.

## Ethical considerations

- **Privacy.** Webcam frames are processed in the browser and never sent anywhere. The page has no server, no analytics, and no storage, and keeps nothing between visits.
- **Data use.** The HaGRID annotations are used under their CC BY-SA 4.0 variant licence, with attribution, and the trained weights carry the same licence. The recorded human games have no licence, so they are used for evaluation only. They are not in the repository, and no shipped model is trained on them beyond the five tuned settings.
- **Low potential for harm.** The models read four gestures and predict moves in a game.
