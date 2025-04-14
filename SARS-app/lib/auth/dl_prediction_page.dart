// dl_prediction_page.dart

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:photo_view/photo_view.dart';
import 'package:photo_view/photo_view_gallery.dart';

class DLPredictionPage extends StatefulWidget {
  @override
  _DLPredictionPageState createState() => _DLPredictionPageState();
}

class _DLPredictionPageState extends State<DLPredictionPage> {
  File? _image;
  final ImagePicker _picker = ImagePicker();
  String _result = '';
  final List<String> groundTruths = [
    'Ground Truth 1',
    'Ground Truth 2',
    'Ground Truth 3',
  ];

  // Pick an image from the gallery
  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  // Send the image to the Flask server for prediction
  Future<void> _predict() async {
    if (_image == null) return;

    final request = http.MultipartRequest(
      'POST',
    //   Uri.parse('https://relevant-tick-mna-a0f12a9b.koyeb.app/predict'), // Replace with your Flask server URL
    // );
    Uri.parse('http://192.168.0.195:5000/predict'), // Replace with your Flask server URL
    );
    request.files.add(
      await http.MultipartFile.fromPath('image', _image!.path),
    );

    final response = await request.send();
    final responseData = await response.stream.bytesToString();

    setState(() {
      _result = responseData;
    });
  }

  // Show a dialog with ground truth values
  void _showGroundTruths() {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: Text(
            'Ground Truth Values',
            style: TextStyle(color: Colors.white),
          ),
          content: Container(
            width: double.maxFinite,
            child: ListView.builder(
              itemCount: groundTruths.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(
                    groundTruths[index],
                    style: TextStyle(color: Colors.white),
                  ),
                );
              },
            ),
          ),
          backgroundColor: Colors.black,
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(),
              child: Text(
                'Close',
                style: TextStyle(color: Colors.blueAccent),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('DL Prediction'),
        backgroundColor: Colors.black,
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Select an Image',
                  style: TextStyle(
                    fontSize: 24,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 20),
                _image == null
                    ? Text(
                  'No image selected.',
                  style: TextStyle(color: Colors.white),
                )
                    : GestureDetector(
                  onTap: () {
                    // Open a zoomable view of the image
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (BuildContext imageContext) => PhotoViewPage(image: _image!),
                      ),
                    );
                  },
                  child: Image.file(
                    _image!,
                    height: 200,
                    fit: BoxFit.cover,
                  ),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _pickImage,
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.blueAccent),
                    shape: MaterialStateProperty.all(
                      RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    padding: MaterialStateProperty.all(
                      EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                    ),
                  ),
                  child: Text(
                    'Pick Image',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                    ),
                  ),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _predict,
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.greenAccent),
                    shape: MaterialStateProperty.all(
                      RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    padding: MaterialStateProperty.all(
                      EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                    ),
                  ),
                  child: Text(
                    'Predict',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                    ),
                  ),
                ),
                SizedBox(height: 20),
                // ElevatedButton(
                //   onPressed: _showGroundTruths,
                //   style: ButtonStyle(
                //     backgroundColor: MaterialStateProperty.all(Colors.orangeAccent),
                //     shape: MaterialStateProperty.all(
                //       RoundedRectangleBorder(
                //         borderRadius: BorderRadius.circular(30),
                //       ),
                //     ),
                //     padding: MaterialStateProperty.all(
                //       EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                //     ),
                //   ),
                //   child: Text(
                //     'Show Ground Truths',
                //     style: TextStyle(
                //       color: Colors.white,
                //       fontSize: 20,
                //     ),
                //   ),
                // ),
                SizedBox(height: 30),
                Text(
                  'Prediction Result:',
                  style: TextStyle(
                    fontSize: 20,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 10),
                Text(
                  _result.isNotEmpty ? _result : 'No result yet.',
                  style: TextStyle(
                    fontSize: 18,
                    color: Colors.white,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// PhotoViewPage class for zooming into the image
class PhotoViewPage extends StatelessWidget {
  final File image;

  const PhotoViewPage({required this.image});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: PhotoView(
          imageProvider: FileImage(image),
          minScale: PhotoViewComputedScale.contained,
          maxScale: PhotoViewComputedScale.covered * 2,
          heroAttributes: const PhotoViewHeroAttributes(tag: "imageHero"),
        ),
      ),
    );
  }
}
