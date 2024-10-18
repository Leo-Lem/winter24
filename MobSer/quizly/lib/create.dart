import 'package:flutter/material.dart';

class CreateWidget extends StatefulWidget {
  const CreateWidget({super.key});

  @override
  State<StatefulWidget> createState() {
    return CreateState();
  }
}

class CreateState extends State<CreateWidget> {
  int currentQuestionIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const TextField(
          decoration: InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'Title',
          ),
        ),
        const Spacer(),
        const TextField(
          decoration: InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'Question?',
          ),
        ),
        const Spacer(),
        ElevatedButton.icon(
            onPressed: () {},
            icon: const Icon(Icons.image),
            label: const Text('Add theme')),
        const Spacer(),
        Text('Question ${currentQuestionIndex + 1} of ?',
            style: Theme.of(context).textTheme.headlineLarge),
        ...(List.generate(4, (index) {
          return Padding(
              padding: const EdgeInsets.all(8.0),
              child: TextField(
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Answer ${index + 1}',
                ),
              ));
        })),
        ElevatedButton(
          onPressed: () {
            setState(() {
              currentQuestionIndex++;
            });
          },
          child: const Text('Next Question'),
        ),
      ],
    );
  }
}
