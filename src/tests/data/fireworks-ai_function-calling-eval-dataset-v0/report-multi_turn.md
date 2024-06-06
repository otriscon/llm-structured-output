case | mlx-community/Meta-Llama-3-8B-Instruct-4bit | gpt-4o-2024-05-13
--- | --- | ---
0 | ✅ | ✅
1 | ✅ | ✅
2 | ✅ | ✅
3 | ✅ | ✅
4 | ✅ | ✅
5 | ✅ | ✅
6 | ✅ | _function_call[0].url_ ~~'www.mywebsite.com'~~ 'http://www.mywebsite.com'
7 | ✅ | ✅
8 | ✅ | ✅
9 | ✅ | ✅
10 | ✅ | ✅
11 | ✅ | ✅
12 | ✅ | ✅
13 | ✅ | ✅
14 | ✅ | ✅
15 | ✅ | ✅
16 | ✅ | ✅
17 | ✅ | ✅
18 | ➕ _function_call[0].genre_ 'action' | ✅
19 | ✅ | ✅
20 | ✅ | ✅
21 | ✅ | ✅
22 | ✅ | ✅
23 | ✅ | ✅
24 | ✅ | ✅
25 | ➕ _function_call[0].genre_ 'pop' ⸱ _function_call[0].keyword_ ~~'pop'~~ 'Taylor Swift' | ✅
26 | ✅ | _function_call[0].country_ ~~'US'~~ 'us'
27 | ✅ | ✅
28 | ✅ | ✅
29 | _function_call[0].amount_ ~~1500 [int]~~ 1500.0 [float] | ✅
30 | ✅ | ✅
31 | ✅ | ✅
32 | ✅ | ✅
33 | ✅ | ✅
34 | ✅ | ✅
35 | ✅ | ✅
36 | _function_call[0].rating_ ~~7 [int]~~ 7.0 [float] | ✅
37 | _function_call[0].date_range['start_date']_ ~~'2022-02-01'~~ '2023-02-20' ⸱ _function_call[0].date_range['end_date']_ ~~'2022-02-08'~~ '2023-02-27' | _function_call[0].date_range['start_date']_ ~~'2022-02-01'~~ '2023-09-30' ⸱ _function_call[0].date_range['end_date']_ ~~'2022-02-08'~~ '2023-10-07'
38 | ✅ | ✅
39 | ✅ | ✅
40 | _function_call[0].event_date_ ~~'2022-04-15'~~ 'today' | _function_call[0].event_date_ ~~'2022-04-15'~~ '2023-10-06'
41 | ✅ | ✅
42 | ✅ | ✅
43 | ✅ | ✅
44 | ✅ | ✅
45 | ✅ | ✅
46 | ✅ | ✅
47 | ✅ | ✅
48 | ✅ | ✅
49 | _function_call[0].query_ ~~''~~ 'comedy' | ➖ ~~_function_call[0].query_ ''~~
50 | ✅ | _function_call[0].url_ ~~'www.example.com'~~ 'http://www.example.com'
51 | ✅ | _function_call[0].username_ ~~'@JohnDoe'~~ 'JohnDoe'
52 | ✅ | ✅
53 | _function_call[0].source_language_ ~~'fr'~~ 'French' ⸱ _function_call[0].target_language_ ~~'en'~~ 'English' | ✅
54 | ✅ | ✅
55 | ✅ | ✅
56 | ✅ | ✅
57 | ✅ | _function_call[0].country_ ~~'United States'~~ 'us'
58 | _function_call[0].event_location_ ~~'conference room in our office'~~ 'Conference room in our office' | _function_call[0].event_date_ ~~'15th of next month'~~ '2023-11-15' ⸱ _function_call[0].event_time_ ~~'10 AM'~~ '10:00 AM' ⸱ _function_call[0].event_location_ ~~'conference room in our office'~~ 'Conference Room, Office'
59 | ✅ | ✅
60 | ✅ | ✅
61 | ✅ | _function_call[0].locations[0]_ ~~'Brooklyn'~~ 'Brooklyn, NY' ⸱ _function_call[0].locations[1]_ ~~'Manhattan'~~ 'Manhattan, NY' ⸱ _function_call[0].locations[2]_ ~~'Queens'~~ 'Queens, NY' ⸱ _function_call[0].locations[3]_ ~~'Brooklyn'~~ 'Brooklyn, NY'
62 | _function_call[0].image_ ~~'user_image'~~ 'The image you sent' | ➕ _tool_call[0]['error']_ {'error': "Parsing tool_calls: KeyError('tool_calls')", 'completion_message': {'role': 'assistant', 'content': 'Please provide the image of the barcode so I can proceed with scanning it.'}} ⸱ ➖ ~~_function_call[0]._ {'name': 'scan_barcode', 'arguments': {'image': 'user_image'}}~~ ⸱ _tool_call[0]['type']_ ~~'function'~~ 'error'
63 | ✅ | ✅
64 | ✅ | ✅
65 | ✅ | ✅
66 | _function_call[0].meal_ ~~'pizza'~~ 'lunch' ⸱ _function_call[0].date_ ~~'2022-03-01'~~ 'today' | _function_call[0].date_ ~~'2022-03-01'~~ '2023-10-10'
67 | ✅ | ✅
68 | ✅ | ✅
69 | _function_call[0].language_ ~~'English'~~ 'en' | _function_call[0].language_ ~~'English'~~ 'en'
70 | ✅ | ✅
71 | _function_call[0].order_items[0]['product_name']_ ~~'laptop'~~ 'Laptop' | ✅
72 | ✅ | ✅
73 | ✅ | _function_call[0].background_color_ ~~'white'~~ '#FFFFFF' ⸱ _function_call[0].foreground_color_ ~~'black'~~ '#000000'
74 | _function_call[0].items[0]['name']_ ~~'apple'~~ 'apples' ⸱ _function_call[0].items[0]['price']_ ~~0.5~~ 1.0 ⸱ _function_call[0].items[1]['name']_ ~~'orange'~~ 'oranges' ⸱ _function_call[0].items[1]['price']_ ~~0.75~~ 0.5 | _function_call[0].items[0]['price']_ ~~0.5~~ 1.0 ⸱ _function_call[0].items[1]['price']_ ~~0.75~~ 0.5
75 | ✅ | ✅
76 | ✅ | ✅
77 | ✅ | ✅
78 | ✅ | ✅
79 | ✅ | _function_call[0].start_date_ ~~'1st June'~~ '2023-06-01' ⸱ _function_call[0].end_date_ ~~'10th June'~~ '2023-06-10'
80 | ✅ | ✅
81 | ✅ | ✅
82 | ✅ | ✅
83 | ➕ _function_call[0].source_currency_ 'USD' ⸱ ➖ ~~_function_call[0].base_currency_ 'USD'~~ ⸱ _function_call[0].['name']_ ~~'get_currency_conversion_rate'~~ 'convert_currency' | ➕ _function_call[0].source_currency_ 'USD' ⸱ ➖ ~~_function_call[0].base_currency_ 'USD'~~ ⸱ _function_call[0].['name']_ ~~'get_currency_conversion_rate'~~ 'convert_currency'
84 | ✅ | ✅
85 | _function_call[0].location_ ~~'main office'~~ 'our main office' | ✅
86 | ✅ | ✅
87 | ✅ | ✅
88 | ✅ | ✅
89 | ✅ | _tool_call[1]_ ➕ {'type': 'function', 'function': {'name': 'get_news', 'arguments': {'interests': ['sports'], 'location': 'New York'}}} ⸱ _function_call[0].interests[1]_ ➖ ~~'sports'~~
90 | ✅ | ✅
91 | ✅ | ✅
92 | ✅ | ✅
93 | ✅ | ✅
94 | ✅ | ✅
95 | ✅ | ✅
96 | ✅ | ✅
97 | ✅ | ✅
98 | ✅ | ✅
99 | ✅ | ✅
pass | 84 (84.0%) | 82 (82.0%)
