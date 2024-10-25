import 'package:first_app/navigation.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(seedColor: Colors.purple),
            textTheme: const TextTheme(
              displayLarge:
                  TextStyle(fontWeight: FontWeight.bold, color: Colors.purple),
              displayMedium: TextStyle(fontWeight: FontWeight.bold),
              headlineLarge: TextStyle(color: Colors.purple),
            )),
        home: Scaffold(
            appBar: AppBar(title: const Text('Quizly')),
            bottomNavigationBar: const NavigationWidget()));
  }
}
