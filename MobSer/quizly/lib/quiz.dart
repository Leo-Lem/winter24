import 'package:first_app/data/quiz.dart';
import 'package:flutter/material.dart';

class QuizWidget extends StatefulWidget {
  const QuizWidget({super.key});

  @override
  State<StatefulWidget> createState() {
    return QuizState();
  }
}

class QuizState extends State<QuizWidget> {
  Quiz currentQuiz = Quiz.example;
  int currentQuestionIndex = 0;
  int points = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text('Quizly', style: Theme.of(context).textTheme.displayLarge),
        const Spacer(),
        Text(
            currentQuiz.questions.elementAt(
              currentQuestionIndex,
            ),
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.displayMedium),
        const Spacer(),
        Text(
            'Question ${currentQuestionIndex + 1} of ${currentQuiz.questionCount()}',
            style: Theme.of(context).textTheme.headlineLarge),
        const Spacer(),
        ...(currentQuiz.answers.elementAt(currentQuestionIndex).map((answer) {
          return ElevatedButton(
            onPressed: () {
              setState(() {
                if (currentQuiz.answers[currentQuestionIndex][currentQuiz
                        .correctAnswers
                        .elementAt(currentQuestionIndex)] ==
                    answer) {
                  points++;
                }

                currentQuestionIndex =
                    (currentQuestionIndex + 1) % currentQuiz.questionCount();
              });
            },
            child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [Text(answer)]),
          );
        }).toList()),
        const Spacer(),
        Text('Points: $points',
            style: Theme.of(context).textTheme.headlineLarge),
      ],
    );
  }
}
