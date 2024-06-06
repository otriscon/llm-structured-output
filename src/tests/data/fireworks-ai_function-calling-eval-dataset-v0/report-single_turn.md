case | mlx-community/Meta-Llama-3-8B-Instruct-4bit | gpt-4o-2024-05-13
--- | --- | ---
0 | ✅ | ✅
1 | ✅ | ✅
2 | ✅ | ✅
3 | ✅ | ✅
4 | ➕ _function_call[0].limit_ 100 ⸱ ➖ ~~_function_call[0].relationship_ 'siblings'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_comments_on_domain' | _function_call[0].relationship_ ~~'siblings'~~ 'sibling_domains'
5 | _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'referrer_files'~~ 'includes' | _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'referrer_files'~~ 'files'
6 | _function_call[0].relationship_ ~~'communicating_files'~~ 'communicates_with' | ✅
7 | ✅ | ✅
8 | ✅ | ➕ _function_call[0].ip_ '192.0.2.1' ⸱ ➕ _function_call[0].relationship_ 'resolutions' ⸱ ➖ ~~_function_call[0].id_ '192.0.2.1'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_dns_resolution_object'~~ 'vt_get_objects_related_to_ip_address'
9 | _function_call[0].relationship_ ~~'referrer_files'~~ 'has_file' | _function_call[0].relationship_ ~~'referrer_files'~~ 'files'
10 | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain'
11 | ✅ | ➕ _function_call[0].ip_ '203.0.113.0' ⸱ ➕ _function_call[0].relationship_ 'resolutions' ⸱ ➖ ~~_function_call[0].id_ '203.0.113.0'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_dns_resolution_object'~~ 'vt_get_objects_related_to_ip_address'
12 | ✅ | ✅
13 | ✅ | ✅
14 | ✅ | _function_call[0].ip_ ~~'http://www.example.org'~~ '93.184.216.34'
15 | ✅ | ✅
16 | _function_call[0].ip_ ~~'12.234.56.126'~~ '22.242.75.136' | _function_call[0].ip_ ~~'12.234.56.126'~~ '22.242.75.136'
17 | _function_call[0].relationship_ ~~'urls'~~ 'related_to' | ✅
18 | ✅ | ✅
19 | _function_call[0].ip_ ~~'explorerweb.org'~~ 'http://explorerweb.org' | ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_ip_address_report' ⸱ _function_call[0].ip_ ~~'explorerweb.org'~~ 'http://explorerweb.org'
20 | ✅ | ✅
21 | ✅ | ✅
22 | _function_call[0].relationship_ ~~'referrer_files'~~ 'contains' | _function_call[0].relationship_ ~~'referrer_files'~~ 'communicating_files'
23 | ➖ ~~_function_call[0].relationship_ 'downloaded_files'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_domain_report' | ✅
24 | _function_call[0].relationship_ ~~'siblings'~~ 'sibling' | _function_call[0].relationship_ ~~'siblings'~~ 'sibling_domains'
25 | ➕ _function_call[0].limit_ 100 ⸱ ➕ _function_call[0].cursor_ '' ⸱ _function_call[0].x-apikey_ ~~'delta_key'~~ 'your_delta_key' | ✅
26 | ✅ | ✅
27 | ✅ | ➕ _function_call[0].ip_ '44.55.66.77' ⸱ ➕ _function_call[0].relationship_ 'resolutions' ⸱ ➖ ~~_function_call[0].id_ '44.55.66.77'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_dns_resolution_object'~~ 'vt_get_objects_related_to_ip_address'
28 | ➖ ~~_function_call[0].relationship_ 'graphs'~~ ⸱ ➖ ~~_function_call[0].x-apikey_ 'sec_key2'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_votes_on_ip_address' | ✅
29 | ✅ | ✅
30 | ✅ | ✅
31 | ✅ | ✅
32 | ➖ ~~_function_call[0].relationship_ 'urls'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
33 | _function_call[0].relationship_ ~~'communicating_files'~~ 'communicates_with' | ✅
34 | ✅ | ➕ _function_call[0].ip_ '10.0.0.1' ⸱ ➕ _function_call[0].relationship_ 'resolutions' ⸱ ➖ ~~_function_call[0].id_ '10.0.0.1'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_dns_resolution_object'~~ 'vt_get_objects_related_to_ip_address'
35 | ✅ | ✅
36 | ✅ | ✅
37 | ✅ | ✅
38 | ✅ | ✅
39 | ✅ | ✅
40 | ➕ _function_call[0].limit_ 100 | ✅
41 | ✅ | ➕ _function_call[0].domain_ 'mysite.io' ⸱ ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ ➖ ~~_function_call[0].ip_ 'mysite.io'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_domain_report'
42 | _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_objects_related_to_domain' ⸱ _function_call[0].relationship_ ~~'communicating_files'~~ 'communicates_with' | ✅
43 | ✅ | ✅
44 | ✅ | ✅
45 | ✅ | ✅
46 | ➖ ~~_function_call[0].relationship_ 'urls'~~ ⸱ ➖ ~~_function_call[0].x-apikey_ 'gamma_key'~~ ⸱ ➖ ~~_function_call[0].cursor_ 'next_page'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_votes_on_ip_address' | ✅
47 | ✅ | ➕ _function_call[0].domain_ 'samplepage.net' ⸱ ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ ➖ ~~_function_call[0].ip_ 'samplepage.net'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_domain_report'
48 | ✅ | ✅
49 | ➖ ~~_function_call[0].cursor_ 'start_cursor'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'downloaded_files'~~ 'downloaded_from' | ➖ ~~_function_call[0].cursor_ 'start_cursor'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address'
50 | ➖ ~~_function_call[0].relationship_ 'parent'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_domain_report' | ➕ _function_call[0].limit_ 1
51 | ✅ | ✅
52 | _function_call[0].relationship_ ~~'caa_records'~~ 'CAA' | ✅
53 | ✅ | ✅
54 | ✅ | ✅
55 | ✅ | ✅
56 | ✅ | ✅
57 | ✅ | ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_ip_address_report'
58 | ➕ _function_call[0].limit_ 0 ⸱ ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
59 | ✅ | _function_call[0].ip_ ~~'https://www.example.org'~~ '93.184.216.34'
60 | ➖ ~~_function_call[0].relationship_ 'siblings'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | _function_call[0].relationship_ ~~'siblings'~~ 'sibling_domains'
61 | _function_call[0].ip_ ~~'viewpage.net'~~ 'http://viewpage.net' | ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_ip_address_report' ⸱ _function_call[0].ip_ ~~'viewpage.net'~~ 'http://viewpage.net'
62 | ✅ | ✅
63 | ✅ | ✅
64 | ➖ ~~_function_call[0].relationship_ 'caa_records'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_domain_report' | ✅
65 | ✅ | _tool_call[1]_ ➕ {'type': 'function', 'function': {'name': 'vt_get_comments_on_domain', 'arguments': {'domain': 'reddit.com', 'x-apikey': 'reddit_api_key'}}}
66 | ✅ | ✅
67 | ✅ | ✅
68 | _function_call[0].relationship_ ~~'historical_whois'~~ 'whois' ⸱ _function_call[0].x-apikey_ ~~'elite_api'~~ 'your_api_key' | _function_call[0].relationship_ ~~'historical_whois'~~ 'whois'
69 | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain'
70 | ✅ | ✅
71 | ➕ _function_call[0].limit_ 100 | ✅
72 | ✅ | ✅
73 | ✅ | ✅
74 | ✅ | ✅
75 | ✅ | ✅
76 | _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_object_descriptors_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'related_threat_actors'~~ 'threat_actor' | _function_call[0].relationship_ ~~'related_threat_actors'~~ 'threat_actors'
77 | ➖ ~~_function_call[0].relationship_ 'subdomains'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
78 | ➖ ~~_function_call[0].relationship_ 'urls'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
79 | ➕ _function_call[0].id_ 'dns_resolution_object_id' ⸱ ➖ ~~_function_call[0].domain_ 'site5.info'~~ ⸱ ➖ ~~_function_call[0].relationship_ 'resolutions'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_dns_resolution_object' | ✅
80 | ✅ | ✅
81 | ✅ | ✅
82 | ✅ | ✅
83 | ✅ | ✅
84 | _function_call[0].relationship_ ~~'historical_whois'~~ 'whois' | ✅
85 | ➕ _function_call[0].id_ 'yahoo.com' ⸱ ➖ ~~_function_call[0].domain_ 'yahoo.com'~~ ⸱ ➖ ~~_function_call[0].relationship_ 'resolutions'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_dns_resolution_object' | ✅
86 | _function_call[0].relationship_ ~~'referrer_files'~~ 'contains' | _function_call[0].relationship_ ~~'referrer_files'~~ 'files'
87 | _function_call[0].ip_ ~~'digdeep.io'~~ 'http://digdeep.io' | ➕ _function_call[0].domain_ 'digdeep.io' ⸱ ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ ➖ ~~_function_call[0].ip_ 'digdeep.io'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_domain_report'
88 | ✅ | ✅
89 | _function_call[0].ip_ ~~'surfthis.net'~~ 'http://surfthis.net' | ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_ip_address_report' ⸱ _function_call[0].ip_ ~~'surfthis.net'~~ 'http://surfthis.net'
90 | _function_call[0].relationship_ ~~'communicating_files'~~ 'communicates_with' | ✅
91 | ➖ ~~_function_call[0].relationship_ 'urls'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
92 | _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'historical_ssl_certificates'~~ 'ssl_certificate' | _function_call[0].relationship_ ~~'historical_ssl_certificates'~~ 'ssl_certificates'
93 | _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_object_descriptors_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'historical_ssl_certificates'~~ 'ssl-certificate' | _function_call[0].relationship_ ~~'historical_ssl_certificates'~~ 'ssl_certificates'
94 | _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_ip_address'~~ 'vt_get_objects_related_to_ip_address' ⸱ _function_call[0].relationship_ ~~'referrer_files'~~ 'REFERENCES' | ✅
95 | _function_call[0].id_ ~~'10.10.10.10linked.site'~~ '10.10.10.10_linked.site' | ✅
96 | _function_call[0].ip_ ~~'checkthisout.net'~~ 'http://checkthisout.net' | ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_ip_address_report' ⸱ _function_call[0].ip_ ~~'checkthisout.net'~~ 'http://checkthisout.net'
97 | _function_call[0].domain_ ~~'sample.org'~~ 'sample.com' ⸱ _function_call[0].relationship_ ~~'cname_records'~~ 'dns_resolution' | _function_call[0].domain_ ~~'sample.org'~~ 'sample.com'
98 | _function_call[0].relationship_ ~~'communicating_files'~~ 'file' | ✅
99 | _function_call[0].x-apikey_ ~~'eta_key'~~ 'your_api_key' | ✅
100 | ➖ ~~_function_call[0].relationship_ 'historical_whois'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_ip_address_report' | ➖ ~~_function_call[0].relationship_ 'historical_whois'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_ip_address'~~ 'vt_get_ip_address_report'
101 | ✅ | ✅
102 | ➖ ~~_function_call[0].x-apikey_ 'KEY123'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_ip_address_report'~~ 'vt_get_votes_on_ip_address' | ✅
103 | ✅ | ✅
104 | ✅ | ✅
105 | ✅ | ✅
106 | _function_call[0].ip_ ~~'inspectlink.com'~~ 'http://inspectlink.com' | ➕ _function_call[0].domain_ 'inspectlink.com' ⸱ ➕ _function_call[0].x-apikey_ 'your_api_key_here' ⸱ ➖ ~~_function_call[0].ip_ 'inspectlink.com'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_votes_on_ip_address'~~ 'vt_get_domain_report'
107 | ✅ | ✅
108 | ➖ ~~_function_call[0].relationship_ 'historical_whois'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_domain_report' | ✅
109 | ✅ | ✅
110 | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain' | ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_objects_related_to_domain'~~ 'vt_get_comments_on_domain'
111 | ➕ _function_call[0].limit_ 100 ⸱ ➖ ~~_function_call[0].relationship_ 'comments'~~ ⸱ _function_call[0].['name']_ ~~'vt_get_object_descriptors_related_to_domain'~~ 'vt_get_comments_on_domain' | ✅
pass | 59 (52.68%) | 77 (68.75%)
