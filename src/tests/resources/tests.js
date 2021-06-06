

var inputVals = {0 : "UP",1: "LEFT",2:"DOWN",3:"RIGHT",4:"ACTION",tick:"TICK",undo:"UNDO",restart:"RESTART"};

function testFunction(td) {

}

for (var i=0;i<testdata.length;i++) {
	test(
		testdata[i][0], 
		function(num){
			return function(){
				var td = testdata[num];
				var testcode = td[1][0];
				var testinput=td[1][1];
				var testresult=td[1][2];
				var input="";
				for (var j=0;j<testinput.length;j++) {
					if (j>0) {
						input+=", ";
					}
					input += inputVals[testinput[j]];
				}
				var errormessage =  testcode+"\n\n\ninput : "+input;
				ok(runTest(td[1]),errormessage);
			};
		}(i)
	);
}




for (var i=0;i<errormessage_testdata.length;i++) {
	test(
		"🐛"+errormessage_testdata[i][0], 
		function(num){
			return function(){
				var td = errormessage_testdata[num];
				var testcode = td[1][0];
				var testerrors=td[1][1];
				if (td[1].length!==3){
					throw "Error/Warning message testdata has wrong number of fields, invalid. Accidentally pasted in level recording data?";
				}
				var errormessage =  testcode+"\n\n\ndesired errors : "+testerrors;
				ok(runCompilationTest(td[1]),errormessage);
			};
		}(i)
	);
}