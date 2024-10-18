import 'package:flutter/material.dart';

class SearchWidget extends StatefulWidget {
  const SearchWidget({super.key});

  @override
  State<StatefulWidget> createState() {
    return SearchState();
  }
}

class SearchState extends State<SearchWidget> {
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        SearchAnchor(
          suggestionsBuilder: (context, controller) => [
            const Text('Suggestion 1'),
            const Text('Suggestion 2'),
            const Text('Suggestion 3')
          ],
          builder: (context, controller) {
            return SearchBar(
                leading: const Icon(Icons.search), controller: controller);
          },
        ),
        const Padding(
          padding: EdgeInsets.all(12.0),
          child: Text('Sort By: Trending', style: TextStyle(fontSize: 20)),
        ),
        Text('Search Results',
            style: Theme.of(context).textTheme.headlineLarge),
        Padding(
          padding: const EdgeInsets.all(20.0),
          child: Text(
            'Quiz 1',
            style: TextStyle(
              background: Paint()
                ..color = Colors.purple
                ..strokeWidth = 20
                ..strokeJoin = StrokeJoin.round
                ..strokeCap = StrokeCap.round
                ..style = PaintingStyle.stroke,
              color: Colors.white,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(20.0),
          child: Text(
            'Quiz 2',
            style: TextStyle(
              background: Paint()
                ..color = Colors.purple
                ..strokeWidth = 20
                ..strokeJoin = StrokeJoin.round
                ..strokeCap = StrokeCap.round
                ..style = PaintingStyle.stroke,
              color: Colors.white,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(20.0),
          child: Text(
            'Quiz 3',
            style: TextStyle(
              background: Paint()
                ..color = Colors.purple
                ..strokeWidth = 20
                ..strokeJoin = StrokeJoin.round
                ..strokeCap = StrokeCap.round
                ..style = PaintingStyle.stroke,
              color: Colors.white,
            ),
          ),
        ),
        const Spacer(),
      ],
    );
  }
}
