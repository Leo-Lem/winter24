class Quiz {
  final List<String> questions;
  final List<List<String>> answers;
  final List<int> correctAnswers;

  int questionCount() {
    return questions.length;
  }

  Quiz({
    required this.questions,
    required this.answers,
    required this.correctAnswers,
  });

  static Quiz example = Quiz(questions: [
    'What is your favorite color?',
    'What is your favorite animal?',
    'What is your favorite food?',
    'What is your favorite movie?',
    'What is your favorite song?',
    'What is your favorite book?',
    'What is your favorite game?',
    'What is your favorite sport?',
    'What is your favorite hobby?',
  ], answers: [
    ['Red', 'Blue', 'Green', 'Yellow'],
    ['Dog', 'Cat', 'Bird', 'Fish'],
    ['Pizza', 'Pasta', 'Burger', 'Salad'],
    ['The Matrix', 'The Lord of the Rings', 'Star Wars', 'The Dark Knight'],
    [
      'Bohemian Rhapsody',
      'Stairway to Heaven',
      'Imagine',
      'Smells Like Teen Spirit'
    ],
    ['1984', 'Brave New World', 'Animal Farm', 'Fahrenheit 451'],
    ['Chess', 'Poker', 'Monopoly', 'Risk'],
    ['Football', 'Basketball', 'Baseball', 'Soccer'],
    ['Reading', 'Writing', 'Drawing', 'Painting'],
  ], correctAnswers: [
    1,
    0,
    2,
    3,
    0,
    0,
    0,
    3,
    0,
  ]);
}
