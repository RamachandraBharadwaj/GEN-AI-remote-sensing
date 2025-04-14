import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'sar_colorization_page.dart'; // Import the SAR colorization page
import 'dl_prediction_page.dart';
import 'vit_prediction_page.dart';
import 'flood_detection.dart';

class DashboardPage extends StatefulWidget {
  @override
  _DashboardPageState createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  int _selectedIndex = 0;

  // Updated list of pages to include the SARColorizationPage
  final List<Widget> _pages = [
    DashboardContent(),
    SARColorizationPage(),
    FloodPage(),
    VITPredictionPage(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: AnimatedSwitcher(
        duration: Duration(milliseconds: 300),
        child: _pages[_selectedIndex],
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.image),
            label: 'SAR Colorization',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.analytics),
            label: 'Flood Detection',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.image),
            label: 'Crop classification',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.logout),
            label: 'Logout',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.blueAccent,
        unselectedItemColor: Colors.black,
        selectedLabelStyle: TextStyle(
          fontSize: 9,
          fontWeight: FontWeight.bold,
          color: Colors.black,
        ),
        unselectedLabelStyle: TextStyle(
          fontSize: 9,
          color: Colors.black,
        ),
        backgroundColor: Colors.grey[200], // Light background to highlight black text
        type: BottomNavigationBarType.fixed, // Prevent shifting effect
        onTap: (index) {
          if (index == 4) {
            // Logout functionality
            FirebaseAuth.instance.signOut();
            Navigator.of(context)
                .pushReplacementNamed('/login'); // Navigate back to the login page
          } else {
            _onItemTapped(index);
          }
        },
      ),
    );
  }
}

class DashboardContent extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Welcome, ${user?.email ?? 'User'}!',
              style: TextStyle(
                fontSize: 24,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 20),
            Text(
              'User Details:',
              style: TextStyle(
                fontSize: 18,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'Email: ${user?.email}',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'User ID: ${user?.uid}',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
