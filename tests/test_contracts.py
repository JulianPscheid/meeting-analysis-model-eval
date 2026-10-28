"""Contract and request regressions; all data is invented and no model is invoked."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev

FIXTURES = Path(ev.__file__).parent / 'fixtures'
NEW_INTENTS = [name for name, spec in ev.INTENTS.items() if not spec.get('legacy')]


def fixture(intent, filename):
    return (FIXTURES / intent / filename).read_text(encoding='utf-8')


class Contracts(unittest.TestCase):
    def test_good_and_bad_for_every_job(self):
        for intent in NEW_INTENTS:
            with self.subTest(intent=intent):
                request = json.loads(fixture(intent, 'request.example.json'))
                schema = json.loads(fixture(intent, 'schema.json'))
                good = ev.check_output(intent, fixture(intent, 'output.good.txt'), schema=schema,
                                       messages=request['messages'])
                bad = ev.check_output(intent, fixture(intent, 'output.bad.txt'), schema=schema,
                                      messages=request['messages'])
                self.assertFalse(good['structurally_bad'], good)
                self.assertTrue(bad['structurally_bad'] or bad['reasoning_leak'], bad)
                if request['grammar']:
                    self.assertEqual(request['grammar'], fixture(intent, 'grammar.gbnf'))
                    ev._assert_grammar_rules_defined(request['grammar'])
        print('new intent checkers: passing and failing fixture for all eight jobs')

    def test_languages(self):
        samples = {
            'en': 'The team has a plan and we will review the results with our colleagues before the next meeting.',
            'de': 'Das Team hat einen Plan und wir werden die Ergebnisse mit unseren Kollegen bei der nächsten Sitzung besprechen.',
            'fr': 'Nous avons un plan et nous allons examiner les résultats avec notre équipe pour la prochaine réunion.',
            'es': 'El equipo tiene un plan y nosotros vamos a revisar los resultados con nuestra colega en la próxima reunión.',
            'it': 'Il gruppo ha un piano e noi discuteremo i risultati con la nostra collega per la prossima riunione.',
            'pt': 'O grupo tem um plano e nós vamos rever os resultados com a nossa colega para a próxima reunião.',
            'nl': 'Het team heeft een plan en wij zullen de resultaten met onze collega voor de volgende vergadering bespreken.',
        }
        for lang, text in samples.items():
            with self.subTest(lang=lang):
                detected, confidence = ev.detect_language(text)
                self.assertEqual(detected, lang)
                self.assertGreater(confidence, 0)
        self.assertEqual(ev.detect_language('Mina 431 t1'), (None, 0))
        self.assertEqual(ev.detect_language(''), (None, 0))
        print('language detector: en de fr es it pt nl short samples passed')

    def test_schema_subset_and_json_edges(self):
        schema = {'type':'object', 'required':['rows'], 'properties':{
            'rows':{'type':'array','minItems':1,'maxItems':2,'items':{
                'type':'string','minLength':2,'maxLength':5,'enum':['okay','yes']}}},
            'additionalProperties':False}
        ev._check_schema(schema)
        self.assertEqual(ev.schema_errors({'rows':['okay']}, schema), [])
        for value in ({}, {'rows':[]}, {'rows':['yes']*3}, {'rows':['x']},
                      {'rows':['longer']}, {'rows':[2]}, {'rows':['okay'],'extra':1}):
            self.assertTrue(ev.schema_errors(value,schema), value)
        self.assertTrue(ev.schema_errors(True, {'type':'integer'}))
        self.assertTrue(ev.schema_errors(True, {'enum':[1]}))
        self.assertFalse(ev.schema_errors(1.0, {'type':'number'}))
        self.assertFalse(ev.schema_errors(1.0, {'type':'integer'}))
        self.assertTrue(ev.schema_errors(1.5, {'type':'integer'}))
        self.assertFalse(ev.schema_errors(None, {'type':['string','null']}))
        self.assertFalse(ev._check_json('NaN', {})['json_valid'])
        self.assertFalse(ev._check_json('{broken', {})['json_valid'])
        fenced = ev._check_json('```json\n{"rows":["okay"]}\n```', schema)
        self.assertTrue(fenced['json_valid'] and fenced['code_fence'] and fenced['structurally_bad'])
        for bad in ({'oneOf':[]}, {'type':'unknown'}, {'minItems':-1}, {'additionalProperties':{}}, {'properties':[]}):
            with self.assertRaises(ValueError):
                ev._check_schema(bad)

    def test_speakers_accent_folding_and_labels(self):
        schema = json.loads(fixture('speaker_identification','schema.json'))
        content = '{"speakers":[{"label":"Speaker 1","name":"Remi Duval"}]}'
        messages = [{'role':'user','content':'Speaker 1: Rémi Duval speaks.'}]
        scores = ev.check_output('speaker_identification', content, schema=schema, messages=messages)
        self.assertEqual(scores['unsupported_names'], [])
        self.assertEqual(scores['unknown_labels'], [])
        self.assertTrue(ev.check_output('speaker_identification',content.replace('Speaker 1','Speaker 9'),
                        schema=schema,messages=messages)['unknown_labels'])
        self.assertTrue(ev.check_output('speaker_identification',content.replace('Remi Duval','Remi'),
                        schema=schema,messages=[{'role':'user','content':'Speaker 1: Notremi speaks.'}])['unsupported_names'])

    def test_cleanup_configurable_paths_and_flat_mode(self):
        schema = {'type':'object','x-eval':{'cleanup':{
            'input_turns':'/messages/0/content/raw~1turns', 'output_turns':'/clean',
            'id':'/key','text':'/words', 'output_id':'/turn','output_text':'/sentence'}}}
        messages = [{'role':'user','content':json.dumps({'raw/turns':[
            {'key':4,'words':'The test is not done'}, {'key':8,'words':'We will wait'}]})}]
        good = '{"clean":[{"turn":4,"sentence":"The test is not done."},{"turn":8,"sentence":"We will wait."}]}'
        scores = ev.check_output('transcript_cleanup',good,schema=schema,messages=messages)
        self.assertEqual((scores['missing_ids'],scores['extra_ids'],scores['duplicate_ids']),([],[],[]))
        self.assertEqual(scores['overall_overlap'],1)
        self.assertEqual(scores['min_turn_overlap'],1)
        self.assertEqual(scores['length_ratio'],1)
        bad = '{"clean":[{"turn":4,"sentence":"Tomorrow."},{"turn":4,"sentence":"Tomorrow."},{"turn":9,"sentence":"Sun."}]}'
        scores = ev.check_output('transcript_cleanup',bad,schema=schema,messages=messages)
        self.assertEqual(scores['missing_ids'],[8])
        self.assertEqual(scores['extra_ids'],[9])
        self.assertEqual(scores['duplicate_ids'],[4])
        self.assertEqual(scores['min_turn_overlap'],0)
        flat_request=json.loads(fixture('transcript_cleanup','request.flat.example.json'))
        flat_schema=json.loads(fixture('transcript_cleanup','schema.flat.json'))
        good=ev.check_output('transcript_cleanup',fixture('transcript_cleanup','output.flat.good.txt'),
                            schema=flat_schema,messages=flat_request['messages'])
        bad=ev.check_output('transcript_cleanup',fixture('transcript_cleanup','output.flat.bad.txt'),
                           schema=flat_schema,messages=flat_request['messages'])
        self.assertFalse(good['structurally_bad'])
        self.assertLess(good['overall_overlap'],1)  # removing a filler lowers this proxy
        self.assertNotIn('missing_ids',good)
        self.assertTrue(bad['structurally_bad'])

    def test_xml_and_text_edges(self):
        schema=json.loads(fixture('highlight_analysis','schema.json'))
        for content in ('<highlight>', '<wrong><meaning/><evidence/><follow_up/></wrong>',
                        '<highlight><meaning/><evidence/><follow_up/><extra/></highlight>',
                        '<!DOCTYPE a [<!ENTITY x "expand">]><highlight/>'):
            self.assertTrue(ev.check_output('highlight_analysis',content,schema=schema)['structurally_bad'])
        for content in ('', '## Title\nFacts', '+ bullet', '1. list'):
            self.assertTrue(ev.check_output('transcript_summary',content)['structurally_bad'])
        self.assertFalse(ev.check_output('chat','## Answer\n- Facts')['structurally_bad'])
        self.assertTrue(ev.check_output('chat','<think>Plan</think> Answer')['reasoning_leak'])

    def test_verdict_enum_auto_discovery(self):
        schema={'type':'object','properties':{'evaluation':{'type':'object','properties':{'decision':{'enum':['show','hide']}}}}}
        scores=ev.check_output('feedback_evaluation','{"evaluation":{"decision":"hide"}}',schema=schema)
        self.assertEqual(scores['verdict'],'hide')
        schema={'type':'object','properties':{'verdict':{'enum':['show','hide']},'strength':{'enum':['low','high']}}}
        self.assertEqual(ev.check_output('feedback_evaluation','{"verdict":"hide","strength":"low"}',schema=schema)['verdict'],'hide')


class Requests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.request=self.root/'request.json'
        self.cases=[{'label':'case one','maxTokens':81,'grammar':None,'messages':[
            {'role':'system','content':'Use supplied evidence.'}, {'role':'user','content':'The plan is ready.'},
            {'role':'assistant','content':'I have read it.'}, {'role':'user','content':'What is ready?'}], 'stream':True},
            {'label':'case two','maxTokens':93,'grammar':'root ::= "ok"','messages':[{'role':'user','content':'Reply ok.'}]}]
        self.write()
        self.results=self.root/'records.jsonl'

    def write(self):
        self.request.write_text(json.dumps(self.cases,ensure_ascii=False)+'\n',encoding='utf-8')

    def argv(self):
        return ['run','--request-file',str(self.request),'--intent','chat','--server',str(self.root/'server'),
            '--model',str(self.root/'model.gguf'),'--label','arm','--out',str(self.root/'out'),
            '--n-ctx','4096','--batch','512','--swa-full','off','--temperature','0.7',
            '--top-p','0.8','--top-k','20','--presence-penalty','1.5','--trials','2']

    def record(self,case=0,**extra):
        value={'input_mode':'request_file','case_label':self.cases[case]['label'],
               'request_messages':self.cases[case]['messages'],'prompt_sha256':ev._file_sha256(self.request),
               'label':'arm','intent':'chat','content':'The plan is ready and we will discuss the next steps with our team.',
               'finish':'stop','reasoning_chars':0,'grammar':'off','error':None}
        value.update(extra)
        return value

    def score(self,records,**flags):
        self.results.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        args=ev.build_parser().parse_args(['score','--request-file',str(self.request),str(self.results)])
        for key,value in flags.items():setattr(args,key,value)
        printed=io.StringIO()
        with contextlib.redirect_stdout(printed): ev.command_score(args)
        return json.loads(printed.getvalue())

    def test_score_refuses_mismatch_and_missing_request(self):
        row=self.record()
        output=self.score([row])
        self.assertFalse(output['runs'][0]['structurally_bad'])
        self.request.write_text(self.request.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'prompt_sha256 does not match --request-file'):
            self.score([row])
        with self.assertRaisesRegex(ValueError,'matching --request-file'):
            ev._record_request(row,None,{},None)
        self.write()
        row['request_messages']=[]
        with self.assertRaisesRegex(ValueError,'stored request_messages'):
            self.score([row])
        print('request-file score: mismatched bytes and messages refused')

    def test_common_flags_and_grouping(self):
        rows=[self.record(),self.record(1,finish='length'),self.record(reasoning_chars=50),
              self.record(content='<think>Plan</think> Answer')]
        output=self.score(rows)
        self.assertEqual([r['structurally_bad'] for r in output['runs']],[False,True,True,True])
        self.assertEqual(set(output['grouped_summary']['chat']),{'case one','case two'})
        allowed=self.score([rows[2]],allow_reasoning=True)
        self.assertFalse(allowed['runs'][0]['structurally_bad'])
        language=self.score([rows[0]],expect_language='de')
        self.assertTrue(language['runs'][0]['wrong_language'])
        self.assertFalse(language['runs'][0]['structurally_bad'])
        strict=self.score([rows[0]],expect_language='de',strict_language=True)
        self.assertTrue(strict['runs'][0]['structurally_bad'])
        leak='Keep these eight special instruction words entirely out of the answer'
        self.cases[0]['messages'][0]['content']=leak;self.write()
        self.assertTrue(self.score([self.record(content=leak)])['runs'][0]['prompt_leaks'])

    def test_request_validation_and_overrides(self):
        args=ev.build_parser().parse_args(self.argv())
        _,_,_,cases=ev._run_cases(args)
        self.assertEqual(cases[0]['messages'],self.cases[0]['messages'])
        self.assertEqual([c['maxTokens'] for c in cases],[81,93])
        args.max_tokens=200;args.grammar='off'
        _,_,_,cases=ev._run_cases(args)
        self.assertEqual([c['maxTokens'] for c in cases],[200,200])
        self.assertEqual([c['grammar'] for c in cases],[None,None])
        grammar=self.root/'override.gbnf';grammar.write_text('root ::= "yes"')
        args.grammar='on';args.grammar_file=str(grammar)
        self.assertTrue(all(c['grammar']==grammar.read_text() for c in ev._run_cases(args)[3]))
        for change in ({'maxTokens':True},{'grammar':4},{'stream':1},{'messages':[]},
                       {'messages':[{'role':'tool','content':'x'}]},{'messages':[{'role':[],'content':'x'}]},{'label':''}):
            bad={**self.cases[0],**change};self.request.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):ev.load_requests(str(self.request))
        self.request.write_text(json.dumps([self.cases[0],self.cases[0]]))
        with self.assertRaisesRegex(ValueError,'unique'):ev.load_requests(str(self.request))

    def test_request_key_and_verbatim_run(self):
        (self.root/'server').write_bytes(b'server');(self.root/'model.gguf').write_bytes(b'gguf')
        args=ev.build_parser().parse_args(self.argv()+['--key-only'])
        def key():
            printed=io.StringIO()
            with contextlib.redirect_stdout(printed): ev.command_run(args)
            return json.loads(printed.getvalue())
        with mock.patch.object(ev,'_host',return_value={'platform':'test'}), mock.patch.object(ev,'_command_text',return_value='version: test'):
            original=key();self.assertEqual(original,key())
            for field,value in [('maxTokens',82),('grammar','root ::= "x"'),('messages',[{'role':'user','content':'changed'}])]:
                before=copy.deepcopy(self.cases)
                self.cases[0][field]=value;self.write()
                self.assertNotEqual(original['reuse_key_sha256'],key()['reuse_key_sha256'])
                self.cases=before;self.write()
            self.request.write_text(self.request.read_text()+' ')
            self.assertNotEqual(original['reuse_key_sha256'],key()['reuse_key_sha256']);self.write()
            sent=[]
            def reply(url,body,timeout):
                sent.append(body)
                return {'choices':[{'finish_reason':'stop','message':{'content':'ok'}}],
                        '__verbose': {'generation_settings': ev._sampler_settings(args, body['seed'])}}
            args.key_only=False
            with mock.patch.object(ev.subprocess,'Popen'),mock.patch.object(ev,'_free_port',return_value=12345),mock.patch.object(ev,'_wait_for_completion'),mock.patch.object(ev,'_assert_metal'),mock.patch.object(ev,'_post_json',side_effect=reply):
                self.assertEqual(ev.command_run(args),0)
            self.assertEqual(len(sent),4)
            for index,body in enumerate(sent):
                case=self.cases[index//2]
                self.assertEqual(body['messages'],case['messages'])
                self.assertEqual(body['max_tokens'],case['maxTokens'])
                self.assertFalse(body['stream'])
                self.assertEqual(body['repeat_last_n'],4096)
                self.assertEqual(body['min_p'],0)
                self.assertEqual(body.get('grammar'),case['grammar'])
            records=[json.loads(line) for line in (self.root/'out/arm.runs.jsonl').read_text().splitlines()]
            self.assertEqual([r['case_label'] for r in records],['case one']*2+['case two']*2)
            self.assertTrue(all(r['prompt_sha256']==ev._file_sha256(self.request) for r in records))
            manifest=json.loads((self.root/'out/arm.manifest.json').read_text())
            self.assertEqual(manifest['reuse_key_sha256'],original['reuse_key_sha256'])
            self.assertEqual(len(manifest['generation_settings']),4)
        print('request-file reuse_key: messages, grammar, maxTokens and exact source bytes covered')

    def test_feedback_verdict_distribution_across_trials(self):
        self.cases=[json.loads(fixture('feedback_evaluation','request.example.json'))]
        self.write()
        rows=[self.record(intent='feedback_evaluation',grammar='on',content=json.dumps(
            {'verdict':verdict,'reason':'The question was already answered.'}))
            for verdict in ('hide','show','hide')]
        output=self.score(rows,schema_file=str(FIXTURES/'feedback_evaluation/schema.json'))
        group=output['summary']['feedback_evaluation'][self.cases[0]['label']]['arm']
        self.assertEqual(group['verdict_distribution'],{'hide':2,'show':1})

    def test_judge_full_messages_and_malformed_result(self):
        prompt=ev._judge_prompt('', '', 'answer','chat',request_messages=self.cases[0]['messages'],
                                schema={'type':'string'},expect_language='en')
        payload=json.loads(prompt.split('INPUT:\n',1)[1])
        self.assertEqual(payload['request_messages'],self.cases[0]['messages'])
        self.assertNotIn('arm',payload)
        valid={'unsupported_claims':[],'coverage':{'early':True,'middle':True,'late':True},
               'length_verdict':'appropriate','length_note':'Fits the question.',
               'structural_leaks':[],'todos':[],'contract_errors':[],
               'completeness_note':'Answered the question.','language_note':'English.',
               'language_matches_request':True}
        self.assertIsNone(ev._parse_judgment_text(json.dumps(valid),'chat')[1])
        malformed={**valid,'length_verdict':[]}
        self.assertIsNone(ev._parse_judgment_text(json.dumps(malformed),'chat')[0])
        del valid['language_matches_request']
        self.assertIsNone(ev._parse_judgment_text(json.dumps(valid),'chat')[0])
        self.results.write_text(json.dumps(self.record())+'\n')
        args=ev.build_parser().parse_args(['judge','--request-file',str(self.request),'--out',str(self.root/'judge.json'),str(self.results)])
        with mock.patch.object(ev,'_judge_with_retry',return_value=(None,'invalid judge JSON','bad')) as judge:
            self.assertEqual(ev.command_judge(args),1)
            self.assertIn('request_messages',judge.call_args.args[0])
        output=json.loads((self.root/'judge.json').read_text())
        self.assertEqual(output['summary']['chat']['case one']['arm']['judge_errors'],1)
        self.assertIsNone(output['summary']['chat']['case one']['arm']['unsupported_claims'])
        self.request.write_text(self.request.read_text()+' ')
        with mock.patch.object(ev,'_judge_with_retry') as judge:
            with self.assertRaisesRegex(ValueError,'prompt_sha256'):
                ev.command_judge(args)
            judge.assert_not_called()


if __name__ == '__main__':
    unittest.main()
