import 'package:first_app/create.dart';
import 'package:first_app/quiz.dart';
import 'package:first_app/search.dart';
import 'package:flutter/material.dart';

class NavigationWidget extends StatefulWidget {
  const NavigationWidget({super.key});

  @override
  State<NavigationWidget> createState() => NavigationState();
}

class NavigationState extends State<NavigationWidget> {
  int _selectedIndex = 0;
  static const List<Widget> _widgetOptions = <Widget>[
    QuizWidget(),
    SearchWidget(),
    CreateWidget()
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Navigation App')),
      body: Center(
          child: Padding(
              padding: const EdgeInsets.all(50),
              child: _widgetOptions.elementAt(_selectedIndex))),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.question_mark),
            label: 'Quiz',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: 'Search',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.plus_one),
            label: 'Create',
          ),
        ],
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
      ),
    );
  }
}
